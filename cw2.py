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
        #Surf keypoint detection
        grey_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints_surf, descriptors_surf = surf.detectANDCompute(img,None)

        return keypoints_surf, descriptors_surf

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

    ### Saman ###
        def find_homography(self, matches, keypoints_l, keypoints_r, iterations=1000, reproj_threshold=5.0):
         """
        Fit the best homography model with the RANSAC algorithm.

        This function calculates a homography matrix that aligns keypoints from two images using
        the RANSAC method to robustly handle outliers. This is particularly useful in applications
        such as image stitching where precise alignment is crucial.

        Parameters:
        - matches: List of cv2.DMatch objects representing the matches between keypoints.
        - keypoints_l: List of cv2.KeyPoint objects for the left image's keypoints.
        - keypoints_r: List of cv2.KeyPoint objects for the right image's keypoints.
        - iterations: Number of RANSAC iterations to perform.
        - reproj_threshold: Maximum allowed reprojection error to treat a point pair as an inlier.

        Returns:
        - homography: The 3x3 homography matrix.
        - mask: Mask of inliers used to compute the homography.
        """
        points_l = np.float32([keypoints_l[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points_r = np.float32([keypoints_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(points_r, points_l, cv2.RANSAC, reproj_threshold, maxIters=iterations)
        # Return the computed homography matrix and the inliers mask
        return homography
    

    ### Saman ###
    def warping(self, img_left, img_right, homography, output_size=None):
        """
        Warp two images into a single panorama using a given homography matrix.

        This method calculates the necessary transformations to align and stitch two images together into a seamless panorama.
        It first determines the boundaries of the resulting panorama by transforming the corner points of the right image
        and combines this with the corner points of the left image.

        Parameters:
        - img_left: The left source image.
        - img_right: The right source image.
        - homography: The homography matrix to transform the right image to align with the left.
        - output_size: Optional. Specify the size of the output panorama (width, height).

        Returns:
        - The combined panorama image.
        """
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        corners_left = np.array([
            [0, 0],
            [0, h_left - 1],
            [w_left - 1, h_left - 1],
            [w_left - 1, 0]
        ], dtype=np.float32).reshape(-1, 1, 2)
        corners_right = np.array([
            [0, 0],
            [0, h_right - 1],
            [w_right - 1, h_right - 1],
            [w_right - 1, 0]
        ], dtype=np.float32).reshape(-1, 1, 2)
        warped_corners_right = cv2.perspectiveTransform(corners_right, homography)
        all_corners = np.concatenate((corners_left, warped_corners_right), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        output_img_size = (x_max - x_min, y_max - y_min)
        warped_image = cv2.warpPerspective(img_right, H_translation.dot(homography), output_img_size)
        cv2.imshow("Warped Right Image", warped_image)
        cv2.waitKey(0)
        for y in range(h_left):
            for x in range(w_left):
                if 0 <= x + translation_dist[0] < output_img_size[0] and 0 <= y + translation_dist[1] < output_img_size[
                    1]:
                    warped_image[y + translation_dist[1], x + translation_dist[0]] = img_left[y, x]
        cv2.imshow("Overlayed Left on Warped Right", warped_image)
        cv2.waitKey(0)

        return warped_image
    
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
        blend_width = end_blend - start_blend
        for col in range(start_blend, end_blend):
                alpha = (col - start_blend) / blend_width # Calculate the blend factor for the current column
                # Update img1's column with a weighted sum of img1's and img2's columns
                img1[:, col] = cv2.addWeighted(img1[:, col], 1 - alpha, img2[:, col], alpha, 0)

        return img1 # Return the modified first image

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
        blend_width = end_blend - start_blend
        for col in range(start_blend, end_blend):
            alpha = 1 / (1 + np.exp(-10 * ((col - start_blend) / blend_width - 0.5)))
            img1[:, col] = cv2.addWeighted(img1[:, col], 1 - alpha, img2[:, col], alpha, 0)
        return img1

## Bob ##
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
    result = stitcher.stitch(img_left, img_right, ...)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
