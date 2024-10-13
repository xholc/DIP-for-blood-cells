import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_bit_plane(image, bit):
    # Extract the bit plane
    bit_plane = np.bitwise_and(image, 2**bit)
    # Scale it to 0 or 255
    bit_plane = bit_plane * 255 / 2**bit

    return bit_plane.astype(np.uint8)

# test different types of WBC and display bit plane images
#save bit plane images for further usage
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
            cv2.imwrite(f'{wbc_type}_bit_plane_{channel_name}_{i}.png', bit_plane)
            plt.imshow(bit_plane, cmap='gray')
            np.save(f'{wbc_type}_bit_plane_{channel_name}_{i}.npy', bit_plane)
            plt.axis('off')
            plt.show()

print("Bit plane images for RGB channels have been generated and saved.")


# this section is to display bit plane images can extract most of information, 
# and the bit planes seem influenced by microscope and dye (systemic factor)
# the bit planes of different color channels are determined by previous step

for wbc_type in ['B', 'E', 'L', 'M', 'N']:
    gbrimg = cv2.imread(f'train/{wbc_type}_(15).jpg')
    #original image
    rgbimg= cv2.cvtColor(gbrimg, cv2.COLOR_BGR2RGB)
    
    output={}
    #features considered important to concate
    img_r= np.load(f'/content/{wbc_type}_bit_plane_r_4.npy')
    kr4 = (img_r+1)* (2**4)-1
    img_r2= np.load(f'/content/{wbc_type}_bit_plane_r_5.npy')
    kr5 = (img_r2+1)* (2**5)-1
    img_g= np.load(f'/content/{wbc_type}_bit_plane_g_4.npy')
    kg4 = (img_g+1)* (2**4)-1
    img_g2= np.load(f'/content/{wbc_type}_bit_plane_g_5.npy')
    kg5 = (img_g2+1)* (2**5)-1
    img_b= np.load(f'/content/{wbc_type}_bit_plane_b_4.npy')
    kb4 = (img_b+1)* (2**4)-1
    img_b2= np.load(f'/content/{wbc_type}_bit_plane_b_5.npy')
    kb5 = (img_b2+1)* (2**5)-1

    i=0
    bit_combination=['kr4kg4kb4','kr4kg4kb5','kr4kg5kb4','kr4kg5kb5','kr5kg4kb4','kr5kg4kb5','kr5kg5kb4','kr5kg5kb5']
    for r_bit in [kr4, kr5]:
        for g_bit in [kg4, kg5]:
            for b_bit in [kb4, kb5]:
                i+=1
                result_fig = np.stack((r_bit, g_bit, b_bit), axis=-1)
                title=f'{wbc_type}_'+ bit_combination[i-1]
                output[title]=result_fig


    fig, ax = plt.subplots(nrows=9, figsize=(20, 30))
    ax[0].imshow(rgbimg)
    ax[0].set_title(f'Original_{wbc_type}_cell')
    ax[0].axis('off')

    #fig, axs = plt.subplots(2,4, figsize=(10, 20))
    for i, (title, result) in enumerate(output.items()):
        ax[i+1].imshow(result)
        ax[i+1].set_title(title)
        ax[i+1].axis('off')