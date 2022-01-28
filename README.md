# DLIM

Deep Learning Pix2Pix model for image generation.

Manual Drawing or image to Japanese (18s) drawing.

The code contains scraper to generate the dataset. It also contains scripts to resize the original Japanese printing without detroying the picture


# RESULTS

Input             |  Generated
:-------------------------:|:-------------------------:
![](results/target_0.png)  |  ![](results/pred_0.png)
![](results/target_1.png)  |  ![](results/pred_1.png)
![](results/target_2.png)  |  ![](results/pred_2.png)
![](results/target_3.png)  |  ![](results/pred_3.png)
![](results/target_4.png)  |  ![](results/pred_4.png)
![](results/target_5.png)  |  ![](results/pred_5.png)
![](results/target_6.png)  |  ![](results/pred_6.png)
![](results/target_7.png)  |  ![](results/pred_7.png)
![](demo/buildings.jpg)  |  ![](demo/buildings_res.jpg)
![](demo/eiffel.jpg)  |  ![](demo/eiffel_res.jpg)
![](demo/samourai.jpg)  |  ![](demo/samourai_res.jpg)


# Dataset / Pretrained weights

The dataset ccan be found in the following link and the pretrained weights for the models.

https://drive.google.com/drive/folders/1iVbeJLTqT5NTXkpj-LnBfecVP41YhX3a?usp=sharing


# Launch

Before launching the algorithm you need to have pretrain weights from the generator:

```
model/gen.pth.tar
```

Moreover, you need to have installed the dependencies detailed in the requirements.txt

To launch the model on a new image:

```
python src/gen_paintings.py <path_to_img>
```

Two images will be generated:

- edges.jpg (Containing the edges extraction with Canny algorithm)
- save.png (Containing the generated printing in japanese style)
