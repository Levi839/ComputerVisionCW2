import numpy as np
import cv2


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, img_left, img_right, ...):  # Add input arguments as you deem fit
        '''
            The main method for stitching two images
        '''

        # Step 1 - extract the keypoints and features with a suitable feature
        # detector and descriptor
        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)

        # Step 2 - Feature matching. You will have to apply a selection technique
        # to choose the best matches
        matches = self.matching(keypoints_l, keypoints_r,
                                descriptors_l, descriptors_r, ...)  # Add input arguments as you deem fit

        print("Number of matching correspondences selected:", len(matches))

        # Step 3 - Draw the matches connected by lines
        self.draw_matches(img_left, img_right, matches)

        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches)

        # Step 5 - Warp images to create the panoramic image
        # Add input arguments as you deem fit
        result = self.warping(img_left, img_right, homography, ...)

        return result

    ### Levi ###
    def compute_descriptors(self, img):
        '''
        The feature detector and descriptor
        '''

        # Your code here

        return keypoints, features

    def matching(keypoints_l, keypoints_r, descriptors_l, descriptors_r, ...):
        # Add input arguments as you deem fit
        '''
            Find the matching correspondences between the two images
        '''

        # Your code here. You should also implement a step to select good matches.

        return good_matches

    def draw_matches(self, img_left, img_right, matches):
        '''
            Connect correspondences between images with lines and draw these
            lines 
        '''

        # Your code here

        cv2.imshow('correspondences', img_with_correspondences)
        cv2.waitKey(0)

    ### Bob ###
        def find_homography(self, matches, keypoints_l, keypoints_r, iterations=1000, reproj_threshold=5.0):
         """
        Fit the best homography model with the RANSAC algorithm.
        """
       
    # Saman #
    def blend_images(self, panorama, img_left, img_right, homography, blend_mode='linear'):
        # Blend the warped right image with the left image using the specified blending mode.
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        inv_homography = np.linalg.inv(homography)
        top_left = np.dot(inv_homography, np.array([0, 0, 1]))
        top_left /= top_left[2]
        bottom_right = np.dot(inv_homography, np.array([w_right, h_right, 1]))
        bottom_right /= bottom_right[2]
        start_blend = int(max(0, top_left[0]))
        end_blend = int(min(w_left, bottom_right[0]))
        if blend_mode == 'linear':
            return self.Blender().linear_blending(panorama, img_left, start_blend, end_blend)
        elif blend_mode == 'custom':
            return self.Blender().customised_blending(panorama, img_left, start_blend, end_blend)
        else:
            return panorama

    ### Saman ###
    def warping(self, img_left, img_right, homography, output_size=None):
        """
        Manually warp the right image using a provided homography matrix and combine it with the left image.

        Returns:
        - The combined panorama image.
        """
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        panorama_width = w_left + w_right
        panorama_height = max(h_left, h_right)
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=img_left.dtype)
        panorama[0:h_left, 0:w_left] = img_left
        for y in range(h_right):
            for x in range(w_right):
                homog_coords = np.dot(homography, np.array([x, y, 1]))
                homog_coords /= homog_coords[2]  # Normalise to convert from homogeneous to Cartesian coordinates
                x_p, y_p = int(homog_coords[0]), int(homog_coords[1])
                if 0 <= x_p < panorama_width and 0 <= y_p < panorama_height:
                    panorama[y_p, x_p] = img_right[y, x]
        return panorama
    

    def remove_black_border(self, img):
        '''
        Remove black border after stitching
        '''
        return cropped_image


### Saman ###
class Blender:
    def linear_blending(self, img1, img2, start_blend, end_blend):
        """
        Linear blending (also known as feathering) across the overlap of two images.

        Parameters:
        img1 (np.array): The first image in which the blend will be applied.
        img2 (np.array): The second image from which pixels will be used in the blend.
        start_blend (int): Start column index for blending in the images.
        end_blend (int): End column index for blending in the images.
        
        Returns:
        np.array: The first image with the blended region modified.
        """

        # Calculate the width of the blending zone
        blend_width = end_blend - start_blend

        # Perform blending from start_blend to end_blend
        for col in range(start_blend, end_blend):
                alpha = (col - start_blend) / blend_width # Calculate the blend factor for the current column
                # Update img1's column with a weighted sum of img1's and img2's columns
                img1[:, col] = cv2.addWeighted(img1[:, col], 1 - alpha, img2[:, col], alpha, 0)

        return img1 # Return the modified first image
    
        # return linear_blending_img

    ### Saman ###
    def customised_blending(self, img1, img2, start_blend, end_blend):
        """
        Perform custom blending using a sigmoidal curve for smooth transition between two images.

        Parameters:
        img1 (np.array): The first image where the blending is applied; this image is modified in-place.
        img2 (np.array): The second image used for blending.
        start_blend (int): The column index to start blending.
        end_blend (int): The column index to end blending.

        Returns:
        np.array: The modified first image with blended region from the second image.
        """

        # Calculate the width of the blending zone
        blend_width = end_blend - start_blend

        # Iterate over each column within the blending zone
        for col in range(start_blend, end_blend):
            # Calculate the alpha value using a sigmoidal function to achieve a smooth transition
            # The sigmoid function shifts from 0 to 1 across the blend width, centered at the midpoint
            alpha = 1 / (1 + np.exp(-10 * ((col - start_blend) / blend_width - 0.5)))

            # Blend the current column of img1 and img2 using the calculated alpha
            # cv2.addWeighted performs per-element weighted sum of two arrays
            img1[:, col] = cv2.addWeighted(img1[:, col], 1 - alpha, img2[:, col], alpha, 0)

        # Return the modified image with blended regions
        return img1
        # return customised_blending_img


class Homography:
    def solve_homography(self, S, D):
        '''
        Find the homography matrix between a set of S points and a set of
        D points
        '''

        # Your code here. You might want to use the DLT algorithm developed in cw1.

        return H


if __name__ == "__main__":

    ### Saman ###
    # Read the image files
    img_left = cv2.imread('s1.jpg')
    img_right = cv2.imread('s2.jpg')

    stitcher = Stitcher()
    # Add input arguments as you deem fit
    result = stitcher.stitch(img_left, img_right, ...)

    # show the result
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
