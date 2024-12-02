import os
import shutil
import sys
import numpy as np
from openslide import open_slide
from shapely.geometry import Polygon
from geojson import Feature, FeatureCollection
import cv2
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from PIL import Image
import pyvips
from tqdm import tqdm
from models.tia_toolbox import UNetModel
import torch
# Retrieve WSI path from command-line arguments
wsi_path = sys.argv[1]
temp_wsi_name = os.path.basename(wsi_path).split('.')[0]


# os.makedirs("temp_output", exist_ok=True)

# Define the processed slide output path
processed_wsi_path = f"processed_slides/{temp_wsi_name}_processed.tif"
os.makedirs("processed_slides", exist_ok=True)
# Load the slide
slide = open_slide(wsi_path)

# # # Initialize the semantic segmentation model
# bcc_segmentor = SemanticSegmentor(
#     pretrained_model="fcn_resnet50_unet-bcss",
#     num_loader_workers=16,
#     batch_size=8,
# )
pretrained_weights = "/home/himanshu/Documents/viewer/models/fcn_resnet50_unet-bcss.pth"
tia_segmentor = UNetModel(
    num_input_channels=3, num_output_channels=5, decoder_block=(3, 3)
)
saved_state_dict = torch.load(pretrained_weights, map_location="cpu")
tia_segmentor.load_state_dict(saved_state_dict)
tia_segmentor = tia_segmentor.to("cuda")
def process_tile(slide, x, y, width, height):
    """Processes a single tile, applies segmentation, and overlays contours."""
    tile = slide.read_region((x, y), 0, (width, height)).convert("RGB")
    # tile_copy = np.array(tile).copy()
    tile = tile.resize((2048, 2048))
    tile = np.array(tile)
    tile = tile/255.0
    tile = tile[np.newaxis, ...]
    tile = torch.tensor(tile).permute(0, 3, 1, 2).float()
    # with torch.no_grad():
    #     tia_segmentor.eval() ??
    tile = tile.to("cuda")
    mask = tia_segmentor(tile)
    mask = mask.squeeze(0)
    mask = mask.cpu().detach().numpy()
    tile = tile.cpu().detach().numpy()
    tile = tile.squeeze(0)
    tile = np.transpose(tile, (1, 2, 0))
    tile = (tile * 255).astype(np.uint8)
    # tile = Image.fromarray(tile)
    # tile.save(f"temp_output/{temp_wsi_name}_tile_{x}_{y}.png")
    # tile = np.array(tile)
    # make a dictionary for classes and their corresponding colors
    #0 is tumor, 1 is stroma, 2 is inflam, 3 is necrosis, 4 is others
    class_colors = {
        0: [255, 0, 0],    # red for tumor
        1: [0, 255, 0],    # green for stroma
        2: [0, 0, 255],    # blue for inflammation
        3: [255, 255, 0],  # yellow for necrosis
        4: [0, 255, 255],  # cyan for others
    }

    mask = np.argmax(mask, axis=0)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    for class_id, color in class_colors.items():
        binary_mask = (mask == class_id).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # in each contour, remove points if they are at the boundary of the tile
        for contour in contours:
            contour = contour.squeeze()
            contour = contour[~np.any((contour == 0) | (contour == width - 1), axis=1)]
            contour = contour[~np.any((contour == 0) | (contour == height - 1), axis=1)]
            if len(contour) > 2:
                cv2.polylines(tile, [contour], isClosed=True, color=color, thickness=10)
        # cv2.drawContours(tile_copy, contours, -1, color, 10)

    # #for the time being, just draw a circle in the middle of the tile
    # tile = cv2.circle(tile, (width // 2, height // 2), 800, (0, 255, 0), 100)
    # tile = Image.fromarray(tile)
    # os.makedirs("temp_output", exist_ok=True)
    # tile.save(f"temp_output/{temp_wsi_name}_tile_{x}_{y}.png")

    return tile

def extract_contours_from_mask(mask):
    """Extracts contours from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.squeeze().tolist() for contour in contours if len(contour) > 2]

# Create an empty slide using pyvips for assembling processed tiles
empty_wsi = pyvips.Image.black(slide.dimensions[0], slide.dimensions[1])

tile_size = 2048  # Define tile size
for y in tqdm(range(0, slide.dimensions[1], tile_size)):
    for x in range(0, slide.dimensions[0], tile_size):
        # print(x, y)
        torch.cuda.empty_cache()
        tile_with_contour = process_tile(slide, x, y, tile_size, tile_size)
        tile_with_contour_vips = pyvips.Image.new_from_array(np.array(tile_with_contour))
        empty_wsi = empty_wsi.insert(tile_with_contour_vips, x, y)
        del tile_with_contour
        del tile_with_contour_vips

# tile_size = 1024  # Define tile size
# for y in tqdm(range(slide.dimensions[1]//4, (slide.dimensions[1]//4)+10000, tile_size)):
#     for x in range(slide.dimensions[0]//4, (slide.dimensions[0]//4)+10000, tile_size):
#         tile_with_contour = process_tile(slide, x, y, tile_size, tile_size)
#         tile_with_contour_vips = pyvips.Image.new_from_array(np.array(tile_with_contour))
#         empty_wsi = empty_wsi.insert(tile_with_contour_vips, x, y)
# Save the processed WSI to the output path
empty_wsi.write_to_file(
    processed_wsi_path,
    compression="jpeg",
    Q=90,
    tile=True,
    tile_width=2048,
    tile_height=2048,
    bigtiff=True,
    pyramid=True,
)
# create a new text file saying that the processing is done
with open(f"processed_slides/{temp_wsi_name}.txt", "w") as f:
    f.write("Processing done")
# Print the location of the processed WSI for the app to pick up
print(f"New WSI saved at {processed_wsi_path}")
