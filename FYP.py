try: 
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
 
    import pandas as pd
    import streamlit as st
    
    import torchvision
    import torch
    import torchvision.transforms as transforms
    import PIL.Image as Image
    import __main__
    import torch.nn as nn 
    import cv2
    import numpy as np
except Exception as e:
    print(e)
 
#STYLE = """
#<style>

#</style>
#"""

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.
    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).
    patch_size : int
        Size of the patch (it is a square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        The emmbedding dimension.
    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2


        self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=1
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """Attention mechanism.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x

class Block(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.network = nn.Sequential(
          #nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=1),
          #nn.BatchNorm2d(16),
          #nn.ReLU(),
          #nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1),
          #nn.BatchNorm2d(32),
          #nn.ReLU(),
          #nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

          nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),

          #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2),padding=1),
          #nn.BatchNorm2d(64),
          #nn.ReLU(),
          #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2),padding=1),
          #nn.BatchNorm2d(64),
          #nn.ReLU(),

          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),

          #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2),padding=1),
          #nn.BatchNorm2d(128),
          #nn.ReLU(),
          #nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(2,2),padding=1),
          #nn.BatchNorm2d(128),
          #nn.ReLU(),
          #nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2),padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(2,2),padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Dropout(0.25)
          #nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

          #nn.Flatten(),
          #nn.Linear(4, num_classes)
          #self.fc1 = nn.Linear(262144, num_classes) # 16*7*7
        )
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same convolution
        #self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 

    def forward(self, x):
        return self.network(x)

class HybridModel(nn.Module):
    """Simplified implementation of the Vision transformer.
    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_chans : int
        Number of input channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            img_size=256,
            patch_size=3,
            in_chans=256,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()
        
        # CNN component
        self.convNet = CNN()

        # ViT compoenents
        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 4, embed_dim)) # self.patch_embed.n_patches
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        
        x = self.convNet(x)
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        
        
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x
    
def load_model(img):
    classes = ['Cherry_(including_sour)___Powdery_mildew', 
     'Corn_(maize)___Common_rust_', 
     'Corn_(maize)___Northern_Leaf_Blight', 
     'Grape___Black_rot', 
     'Grape___Esca_(Black_Measles)', 
     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', #
     'Peach___Bacterial_spot', 
     'Pepper_bell___Bacterial_spot', 
     'Potato___Early_blight', 
     'Potato___Late_blight', 
     'Squash___Powdery_mildew', 
     'Strawberry___Leaf_scorch', 
     'Tomato___Bacterial_spot', 
     'Tomato___Early_blight', 
     'Tomato___Late_blight', 
     'Tomato___Leaf_Mold', 
     'Tomato___Septoria_leaf_spot', 
     'Tomato___Spider_mites Two-spotted_spider_mite', 
     'Tomato___Target_Spot', 
     'Tomato___Yellow_Leaf_Curl_Virus']
    
    # grapes.extension.org
    # extension.umn.edu
    # canr.msu.edu
    # extension.wvu.edu
    # ndsu.edu
    # savvygardening.com
    # content.ces.ncsu.edu
    # audreyslittlefarm.com
    # vegetables.bayer.com
    # ipm.ucanr.edu
    causes = ["Germination and fungal infection via high humidity. Wind may cause the spreading to be more severed.",
        "Rust fungus infection through cool temperatures (60 - 76 degrees F), heavy dews, approximately six hours of leaf wetness, humidity greater than about 95%.",
        "Fungus Exserohilum turcicum (syn. Helminthosporium turcicum) widespreads during moderate temperatures (64 to 80 degrees F) with overexposed moisture periods.",
        "Black rot (Guignardia bidwellii) fungus infection which occurs from mid-bloom until the berries begin to have color. Temperatures of 70°F to 80°F require the shortest period of leaf wetness and are the most suitable temperatures for fungal infection. The fungus is also capable of infection from 50°F to 90°F.",
        "Phaeomoniella chlamydospora and Phaeoacremonium spp. fungi infection which are relatively new genera of fungi that cause the measles and grapevine decline.",
        "",
        "Pathogen Xanthomonas arboricola pv. pruni (XAP) infection.", 
        "Xanthomonas campestris pv. vesicatoria infection through relative humidity above 85%, exposed periods of leaf wetness and heat waves, especially when night temperatures remain above 70°F.",
        "Alternaria tomatophila and Alternaria solani infection which is developed at moderate to warm (59 to 80 F) temperatures; 82 to 86 F is its optimum temperature range and relative humidity is 90'%' or greater.",
        "Fungal-like oomycete pathogen Phytophthora infestans infection which is favored by free moisture and cool to moderate temperatures of 50 to 60 F at night and 60 to 70 F at day.",
        "Erysiphe cichoracearum infection through warm and dry weather. Moist or wet conditions are not needed for the infectants to grow and spread.",
        "Fungus Diplocarpon earlianum infection through leaf wetness during warm weather (68-86°F).",
        "Xanthomonas vesicatoria, X. euvesicatoria, X. gardneri, and X. perforans. are responsible for the caused of this infection under a high temperatures of 75°F to 86°F, high humidity, and frequent rainfall/overhead irrigation.",
        "Alternaria tomatophila and Alternaria solani infection which is developed at moderate to warm (59 to 80 F) temperatures; 82 to 86 F is its optimum temperature range and relative humidity is 90'%' or greater.",
        "Oomycete pathogen Phytophthora infestans (P. infestans) infection through cool, wet weather; clouds protect the spores from exposure to UV radiation by the sun, and wet conditions.",
        "Fungus Passalora fulva infection through an extended periods of leaf wetness and the relative humidity is high (greater than 85%). Can be developed in the early spring temperatures (50.9 degrees Fahrenheit) or those characteristic of summer (90 F). The optimal temperature tomato leaf mold is in the low 70s.",
        "Fungus Septoria lycopersici infection through high humidity and warm temperatures.",
        "Hot and dry conditions. Likely happens in the middle or end of summer. ",
        "Corynespora cassiicola pathogen infection through high humidity levels, periods of leaf wetness from 16 to 44 hours and temperatures between 60° and 90°F. Optimal disease development occurs at temperatures between 68° and 82°F",
        "Caused by Bemisia whitefly species. The virus is primarily spread through the movement of infected plants. The virus can be spread through wind."
    ]
    
    Treatment = ["Try to prevent spreading through leaves by keeping the infected leaves off the plant. Keeping a consistent pace of program from shuck fall via harvesting.",
        "Fungicides are crucial for the treatment especially they are applied during the early stage when few pustules are visible on leaves.",
        "Fungicides could be useful during the early stages. Crop rotation and tillage approaches may be helpful in some cases.",
        "Infected fruits should be disposed to prevent further spread. Fungicides spraying should be scheduled wisely. Planting resistant varieties is encouraged.",
        "No effective treatment strategies for measles. Infected parts are best to be discarded.",
        "",
        "Formulated 53'%' copper could act as a suppression. Establish sod strips between trees and to use gravel or other dust-suppressing methods on nearby dirt roads",
        "Stopping or reducing survival of the pathogen. Resistant varieties like F1 of Autry, Green Flash, Labelle etc",
        "Avoid nitrogen and phosphorus deficiency. Select a late-season variety with a lower susceptibility to early blight. Useful fungicides such as Chlorothalonil, Mancozeb, Azoxystrobin etc",
        "Do not mix seed lots because cutting can transmit late blight. Applying phosphorous acid to potatoes after harvest and before piling can prevent infection and the spread of late blight in storage.",
        "Can be prevented through providing good air circulation by spacing squash plants several feet apart. Cut off any leaves that show early signs of infection ASAP. Combination of baking soda (sodium bicarbonate) with lightweight horticultural oil can help prevention and combating",
        "Selecting a planting site with good air drainage and sun exposure. Allow adequate spacing between them to increase airflow.",
        "Hot water treatment can be used to kill bacteria on and in seed. Avoid over-watering and handle plants as little as possible. Disinfect greenhouses, tools, and equipment between seedling crops with a commercial sanitizer. Streptomycin pesticide is labeled for greenhouse use.",
        "Rotate out of tomatoes and related crops for at least two years. Do not over-fertilize with potassium and maintain adequate levels of both nitrogen and phosphorus. Possible fungicide ingredients are azoxystrobin + chlorothalonil, azoxystrobin + difenoconazole, boscalid etc.",
        "Plant early in the season to escape high disease pressure later in the season. Do not allow water to remain on leaves for long periods of time. Ingredients of fungicides such as Fluopicolide, Oxathiapiprolin + chlorothalonil, Oxathiapiprolin + mandidpropamid etc.",
        "Limiting the relative humidity in the hoophouse can help to prevent this disease. Promote air circulation. Organic preventive products such as Champ (Copper Hydroxide), Double Nickle (Bacillus amyloliquefaciens), Oxidate (Hydrogen dioxide) etc.",
        "Rotate away from tomato for 2 or more years. Remove crop debris from planting areas (particularly tomato crop debris). Ingredients fungicides such as azoxystrobin, chlorothalonil, chlorothalonil + cymoxanil etc.",
        "Early stage can be prevented through spraying the cobweb off with hose water. Natural pesticide like neem oil could be helpful. Repeat treatment every 7 days or so until the spider mites are gone.",
        "Improving airflow through the canopy. Provie wider plant spacing. Avoid over-fertilizing with nitrogen, which can cause overly lush canopy formation. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control.",
        "Plant immediately after any tomato-free period or true winter season. Cover plants with floating row covers of fine mesh (Agryl or Agribon) to protect from whitefly infestations. Use virus- and whitefly-free transplants."
    ]
    
    setattr(__main__, "HybridModel", HybridModel)
    model = torch.load('model.pth', map_location=torch.device("cpu"))

    image_transform = transforms.Compose([
                                            transforms.Resize((224,224)),
                                            transforms.ToTensor()
                                        ])
    
    #model1 = model.eval()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.imread(img)
    image = Image.open(img)
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(opencvImage, (256,256), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    
    image = Image.fromarray(image)
    #image = Image.open(img)
    #im_pil = Image.fromarray(img)
    image = image_transform(image).float()
    image = image.unsqueeze(0)
    
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    
    predicted = classes[predicted.item()]
    cause = causes[classes.index(predicted)]
    treat = Treatment[classes.index(predicted)]
    disease = predicted.split("___")[1].split("_")
    disease_name = ""
    leaf = predicted.split("___")[0].split("_")
    leaf_name = ""
    for i in leaf:
        leaf_name = leaf_name + i + " "
    for j in disease:
        disease_name = disease_name + j + " "

    st.markdown("---")
    st.markdown("<h3 style='text-align: left; color: GreenYellow;'>Plant Condition Summary</h3>", unsafe_allow_html=True)
    
    st.markdown("<h5 style='text-align: left; color: cyan;'>Plant Type: </h5>", unsafe_allow_html=True)
    st.write(leaf_name)
    
    st.markdown("<h5 style='text-align: left; color: cyan;'>Infected Disease: </h5>", unsafe_allow_html=True)
    st.write(disease_name)
    
    st.markdown("<h5 style='text-align: left; color: cyan;'>Possible Causes: </h5>", unsafe_allow_html=True)
    st.write(cause)
    
    st.markdown("<h5 style='text-align: left; color: cyan;'>Possible Treatments & Prevention Tips: </h5>", unsafe_allow_html=True)
    st.write(treat)

def main():
    """Run this function to display the Streamlit app"""
    #st.info(__doc__)
    
    #st.markdown(STYLE, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: GreenYellow;'>Plant Disease Recognition App</h1>", unsafe_allow_html=True)
    
    # Image File Upload
    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()
 
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg"]))
        return
 
    #content = file.getvalue()
 
    if isinstance(file, BytesIO):
        show_file.image(file)
    #print(file)
    load_model(file)
    
    file.close()
    
    
    
main()