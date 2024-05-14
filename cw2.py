import numpy as np
import cv2

class Stitcher:
    def __init__(self):
        pass

    def stitch(self, img_left, img_right, match_threshold=0.75, ransac_iterations=1000, ransac_threshold=5.0,
               panorama_size=None, blend_mode='linear'):
        """
        The main method for stitching two images
        """

        # Step 1 - extract the keypoints and features with a suitable feature
        # detector and descriptor
        keypoints_l, descriptors_l = self.compute_descriptors(img_left)
        keypoints_r, descriptors_r = self.compute_descriptors(img_right)

        # Step 2 - Feature matching. You will have to apply a selection technique
        # to choose the best matches
        matches = self.matching(keypoints_l, keypoints_r, descriptors_l, descriptors_r, match_threshold)
        print("Number of matching correspondences selected:", len(matches))

        # Step 3 - Draw the matches connected by lines
        self.draw_matches(img_left, keypoints_l, img_right, keypoints_r, matches)

        # Step 4 - fit the homography model with the RANSAC algorithm
        homography = self.find_homography(matches, keypoints_l, keypoints_r, ransac_iterations, ransac_threshold)

        # Step 5 - Warp images to create the panoramic image
        result_with_black_borders = self.warping(img_left, img_right, homography, panorama_size)

        ### Saman ###
        # Display the image with black borders before blending and removing them
        cv2.imshow('Stitched Image with Black Borders', result_with_black_borders)
        cv2.waitKey(0)

        # Optional Step 6 - Blend images using the selected mode
        if blend_mode == 'linear':
            result = self.blend_images(result_with_black_borders, img_left, img_right, homography)

        # Optional Step 7 - Remove black borders from the final image
        result = self.remove_black_border(result)
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
        Warp two images into a single panorama using a given homography matrix.
        """
    
    ### Saman ###
    def remove_black_border(self, img):
        '''
        Remove black border after stitching
        '''


### Saman ###
class Blender:
    def linear_blending(self, img1, img2, start_blend, end_blend):
        """
        Linear blending (also known as feathering) across the overlap of two images.
        """

    ### Saman ###
    def customised_blending(self, img1, img2, start_blend, end_blend):
        """
        Perform custom blending using a sigmoidal curve for smooth transition between two images.
        """

## Bob ##
class Homography:
    def solve_homography(self, S, D):
        """
        This method compute the homography matrix using Direct Linear Transformation (DLT).
        S and D are lists of corresponding points (source and destination).
        """
        A = []
        for si, di in zip(S, D):
            x, y = si[0], si[1]
            u, v = di[0], di[1]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        return H / H[2, 2]  # Normalise to make h33 = 1


if __name__ == "__main__":
    ### Saman ###
    # Read the image files
    img_left = cv2.imread('s1.jpg')
    img_right = cv2.imread('s2.jpg')
    stitcher = Stitcher()
    result = stitcher.stitch(img_left, img_right, ...)
    
    # show the result
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
