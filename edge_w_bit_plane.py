import cv2
import get_bit_plane
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage import segmentation
from skimage.segmentation import find_boundaries

# To demonstrate the bit plane containing information to better defined edges
# This step is to determine blur degree in canny edge detector, larger the edge will be smoother
for wbc_type in ['B', 'E', 'L', 'M', 'N']:
    # Read the image in BGR format (OpenCV default)
    img = cv2.imread(f'train/{wbc_type}_(15).jpg')

    # Split the image into BGR channels
    b, g, r = cv2.split(img)
    print(f'Start to get bit_planes of {wbc_type}')
    # Generate and save bit planes for each channel
    channels = {'r': r, 'g': g, 'b': b}
    
    for channel_name, channel in channels.items():
        print(f'{wbc_type}_cell_channel_{channel_name}')
        
        for i in range(8):
            bit_plane = get_bit_plane(channel, i)
            #cv2.imwrite(f'{wbc_type}_bit_plane_{channel_name}_{i}.png', bit_plane)
            edges1 = feature.canny(bit_plane, sigma=0.2)
            edges2 = feature.canny(bit_plane, sigma=0.3)
            edges3 = feature.canny(bit_plane, sigma=0.4)
            edges4 = feature.canny(bit_plane, sigma=0.5)

            # display results
            fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 30))

            ax[0].imshow(bit_plane, cmap='gray')
            ax[0].set_title(f'{wbc_type}_bit_plane_{channel_name}_{i}', fontsize=10)

            ax[1].imshow(edges1, cmap='gray')
            ax[1].set_title(r'Canny filter, $\sigma=0.2$', fontsize=10)

            ax[2].imshow(edges2, cmap='gray')
            ax[2].set_title(r'Canny filter, $\sigma=0.3$', fontsize=10)

            ax[3].imshow(edges3, cmap='gray')
            ax[3].set_title(r'Canny filter, $\sigma=0.4$', fontsize=10)

            ax[4].imshow(edges4, cmap='gray')
            ax[4].set_title(r'Canny filter, $\sigma=0.5$', fontsize=10)

            for a in ax:
                a.axis('off')

            fig.tight_layout()
            plt.show()
            
            
for wbc_type in ['B', 'E', 'L', 'M', 'N']:
    # Read the image in BGR format (OpenCV default)
    img = cv2.imread(f'train/{wbc_type}_(15).jpg')
    img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Split the image into BGR channels
    b, g, r = cv2.split(img)
    print(f'Start to get bit_planes of {wbc_type}')
    #bit plane
    bit_plane = get_bit_plane(g, 4)
    edges = feature.canny(bit_plane, sigma=0.46)
    #finf edges on bit planes
    edge_i= find_boundaries(edges, mode='inner').astype(np.uint8)
    edge_ie= find_boundaries(edges, connectivity=4, mode='inner').astype(np.uint8)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 30))
    ax[0].imshow(img_rgb)
    ax[0].set_title(f'{wbc_type}_bit_plane',fontsize=10)

    ax[1].imshow(segmentation.mark_boundaries(img_rgb, edge_i, color=(1,1,1)))
    #ax3.contour(mask, colors='red', linewidths=1)
    ax[1].set_title('canny',fontsize=10)

    ax[2].imshow(segmentation.mark_boundaries(img_rgb, edge_ie, color=(1,1,1)))
    #x3.contour(mask, colors='red', linewidths=1)
    ax[2].set_title('canny_w_conn',fontsize=10)

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()