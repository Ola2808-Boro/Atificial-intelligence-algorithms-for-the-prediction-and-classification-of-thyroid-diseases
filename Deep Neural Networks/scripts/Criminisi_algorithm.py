
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve


class CriminisiAlgorithm():
    
    def __init__(self,img,mask,patch_size:int,plot_progress:bool):
        self.image=img.astype('uint8')
        self.mask=mask.astype('uint8')
        self.patch_size=patch_size
        self.plot_progress=plot_progress


        #
        self.priority=None
        self.confidence=None
        self.data=None
        self.working_image = None
        self.working_mask = None
        self.front = None

        print(f'Shape img {self.image.shape}, shape mask {self.mask.shape}')
    def inpaint(self):
        self._validate_input()
        self._initialize_attributes()
        start_time=time.time()
        keep_going=True
        while keep_going:
            self._find_front() #find esges in mask
            if self.plot_progress:
                self._plot_image() #
            self._update_priority()
            target_pixel = self._find_highest_priority_pixel() #p
            find_start_time = time.time()
            source_patch = self._find_source_patch(target_pixel)
            print('Time to find best: %f seconds'
                  % (time.time()-find_start_time))
            self._update_image(target_pixel, source_patch)
            keep_going = not self._finished()

        print('Took %f seconds to complete' % (time.time() - start_time))
        return self.working_image


    def _initialize_attributes(self):
        #The confidence is initially the inverse of the mask, that is, the
        #target region is 0 and source region is 1
        #The data starts with zero for all pixels.
        width,height=self.image.shape[:2]
        self.confidence=(1-self.mask).astype(float)
        self.data=np.zeros([width,height])
        self.working_image=np.copy(self.image)
        self.working_mask=np.copy(self.mask)

    def _validate_input(self):
        #mask and image must have the same shape
        if self.mask.shape[:2]!=self.image.shape[:2]:
            raise AttributeError('mask and image must be of the same size')
        
    def _find_front(self):
        #find edges in mask
        self.front = (laplace(self.working_mask) > 0).astype('uint8')
        # print('Front',self.front.shape)
        # plt.imshow(self.front)
        # plt.show()

    def _find_source_patch(self, target_pixel):
        target_patch = self._get_patch(target_pixel)
        height, width = self.working_image.shape[:2]
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        lab_image = rgb2lab(self.working_image)
        for y in range(height - patch_height + 1):
            for x in range(width - patch_width + 1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                if self._patch_data(self.working_mask, source_patch).sum() != 0:
                    continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
        return best_match

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance
    
    def _update_priority(self):
        #P(p)=C(p)*D(p)
        self._update_confidence()
        self._update_data()
        self.priority = self.confidence * self.data * self.front

    def _update_confidence(self):
        new_confidence=np.copy(self.confidence)
        print(f'new_confidence 1 {new_confidence.shape}')
        front_positions=np.argwhere(self.front==1)
        print(f'front_positions shape {front_positions.shape}')
        for point in front_positions:
            patch=self._get_patch(point)
            #print(f'Patch shae {len(patch)}')
            new_confidence[point[0], point[1]] = sum(sum(
                self._patch_data(self.confidence, patch)
            ))/self._patch_area(patch)
        self.confidence = new_confidence
        print(f'new_confidence 2 {self.confidence.shape}')

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        pixels_positions = np.argwhere(
            self._patch_data(
                self.working_mask,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]
        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        mask = self._patch_data(self.working_mask, target_patch)
        rgb_mask = self._to_rgb(mask)
        source_data = self._patch_data(self.working_image, source_patch)
        target_data = self._patch_data(self.working_image, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self._copy_to_patch(
            self.working_image,
            target_patch,
            new_data
        )
        self._copy_to_patch(
            self.working_mask,
            target_patch,
            0
        )

    def _update_data(self):
        normal = self._calc_normal_matrix()
        gradient = self._calc_gradient_matrix()

        normal_gradient = normal*gradient
        self.data = np.sqrt(
            normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
        ) + 0.001  # To be sure to have a greater than 0 data

    def _finished(self):
        height, width = self.working_image.shape[:2]
        remaining = self.working_mask.sum()
        total = height * width
        print('%d of %d completed' % (total-remaining, total))
        return remaining == 0
    
    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)
        return point
    
    def _calc_normal_matrix(self):
        x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.working_mask.astype(float), x_kernel)
        y_normal = convolve(self.working_mask.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal/norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        height, width = self.working_image.shape[:2]

        grey_image = rgb2gray(self.working_image)
        grey_image[self.working_mask == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_val.argmax(),
                patch_gradient_val.shape
            )

            max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_gradient[point[0], point[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_gradient
    
    def _get_patch(self,point):
        half_patch_size = (self.patch_size-1)//2
        height, width = self.working_image.shape[:2]
        patch=[
            [
                max(0,point[0]-half_patch_size),#x1
                min(point[0]+half_patch_size,height-1),#x2
            ],
            [
                max(0, point[1] - half_patch_size),#y1
                min(point[1] + half_patch_size, width-1)#y2
            ]
        ]

        return patch #[[x1,x2],[y1,y2]]

    def _plot_image(self):
        height, width = self.working_mask.shape

        # Remove the target region from the image
        inverse_mask = 1 - self.working_mask
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.working_image * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255

        # Fill the inside of the target region with white
        white_region = (self.working_mask - self.front) * 255
        rgb_white_region = self._to_rgb(white_region)
        image += rgb_white_region

        plt.clf()
        plt.imshow(image)
        plt.draw()
        plt.pause(0.001)  # TODO: check if this is necessary
        width,height=self.working_mask[:2]
        #remove target region from img
        # plt.imshow(self.working_mask)
        # plt.title('Working maska')
        # plt.show()
        # inverse_mask=1-self.working_mask
        # plt.imshow(1-self.working_mask)
        # plt.title('Inverse mask')
        # plt.show()
        # print(f'Uniwge {np.unique(inverse_mask)}')
        # rgb_inverse_mask=self._to_rgb(inverse_mask)
        # #image wihout target region
        # image = self.working_image * rgb_inverse_mask
        # # Fill the target borders with red
        # image[:, :, 0] += self.front * 255
        # plt.imshow(image[:, :, 0])
        # plt.title('Fill the target borders with red')
        # plt.show()
        #  # Fill the inside of the target region with white
        # white_region = (self.working_mask - self.front) * 255
        # plt.imshow(white_region)
        # plt.title('Fill the inside of the target region with white')
        # plt.show()
        # rgb_white_region = self._to_rgb(white_region)
        # print(f'White shape {rgb_white_region.shape}')
        # image += rgb_white_region
        # plt.figure(figsize=(10, 5))
        # # Original Image
        # plt.subplot(1, 2, 1)
        # #Display the original image using matplotlib
        # plt.imshow(rgb_white_region)
        # plt.title('rgb_white_region')
        # plt.axis('off')

        # # Edge-detected Image
        # plt.subplot(1, 2, 2)
        # #Display the edge-detected image using matplotlib with a grayscale color map.
        # plt.imshow(image)
        # plt.title('Image')
        # plt.axis('off')

        # # Show the plot containing both images.
        # plt.show()
        # print(f'Whiet shape rgb{rgb_white_region.shape}, image shape {image.shape}')
        # plt.clf()
        # plt.imshow(image)
        # plt.draw()
        # plt.pause(0.001) 

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape
        return image.reshape(height, width, 1).repeat(3, axis=2)
        
    @staticmethod
    def _patch_data(source, patch):
        #return patch from source region
        return source[patch[0][0]:patch[0][1]+1,patch[1][0]:patch[1][1]+1]
    
    @staticmethod
    def _patch_area(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])
    
    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data