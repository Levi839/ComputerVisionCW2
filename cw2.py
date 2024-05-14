import numpy as np
import cv2


class Stitcher:
    def __init__(self):
        self.homography_solver = self.Homography()

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

        # Display the image with black borders before blending and removing them
        cv2.imshow('Stitched Image with Black Borders', result_with_black_borders)
        cv2.waitKey(0)

        # Optional Step 6 - Blend images using the selected mode
        if blend_mode == 'linear':
            result = self.blend_images(result_with_black_borders, img_left, img_right, homography)

        # Optional Step 7 - Remove black borders from the final image
        result = self.remove_black_border(result)
        return result

    # Levi #
    def compute_descriptors(self, img):
        """
        The feature detector and descriptor
        """
        sift = cv2.SIFT_create()
        keypoints, features = sift.detectAndCompute(img, None)
        return keypoints, features

    # Levi #
    def matching(self,keypoints_l, keypoints_r, descriptors_l, descriptors_r,match_threshold):
        # Add input arguments as you deem fit
        '''
            Find the matching correspondences between the two images
        '''
        matches = []
        for k1, desc_l in enumerate(descriptors_l):
            distances = []
            for k2, desc_r in enumerate(descriptors_r):
                eucli_distance = np.linalg.norm(desc_l - desc_r)
                distances.append((eucli_distance, k2))

            distances.sort(key=lambda x: x[0])

            if len(distances) >= 2 and distances[0][0] < match_threshold * distances[1][0]:
                matches.append(cv2.DMatch(_queryIdx=k1, _trainIdx=distances[0][1], _distance=distances[0][0]))

        return matches

    # Levi #
    def draw_matches(self, img_left, keypoints_l, img_right, keypoints_r, matches):
        '''
            Connect correspondences between images with lines and draw these
            lines
        '''
        # Your code here
        # Draw matches
        img_with_correspondences = cv2.drawMatches(img_left, keypoints_l, img_right, keypoints_r, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Correspondences', img_with_correspondences)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Bob #
    def find_homography(self, matches, keypoints_l, keypoints_r, iterations=1000, reproj_threshold=5.0):
        # Fit the best homography model with the RANSAC algorithm using custom implementation.
        points_l = np.float32([keypoints_l[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        points_r = np.float32([keypoints_r[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        best_H = None
        max_inliers = 0
        for _ in range(iterations):
            indices = np.random.choice(len(matches), 4, replace=False)
            src_points = points_r[indices]
            dst_points = points_l[indices]
            H = self.homography_solver.solve_homography(src_points, dst_points)
            projected_points = self.apply_homography(points_r, H)
            errors = np.linalg.norm(points_l - projected_points, axis=1)
            inliers = errors < reproj_threshold
            num_inliers = np.sum(inliers)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H
        return best_H

    # Bob #
    def apply_homography(self, points, H):
        # Apply homography matrix H to a set of points
        num_points = points.shape[0]
        points_homog = np.hstack((points, np.ones((num_points, 1))))
        transformed_points_homog = np.dot(H, points_homog.T).T
        transformed_points = transformed_points_homog[:, :2] / transformed_points_homog[:, 2][:, np.newaxis]
        return transformed_points

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

    # Saman #
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

    # Saman #
    def remove_black_border(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            img = img[y:y + h, x:x + w]
        return img

    # Saman #
    class Blender:
        def linear_blending(self, panorama, img, start_blend, end_blend):
            blend_width = end_blend - start_blend
            for col in range(start_blend, end_blend):
                alpha = (col - start_blend) / float(blend_width)
                panorama[:, col] = cv2.addWeighted(panorama[:, col], 1 - alpha, img[:, col], alpha, 0)
            return panorama

        # Saman #
        def customised_blending(self, panorama, img, start_blend, end_blend):
            blend_width = end_blend - start_blend
            for col in range(start_blend, end_blend):
                alpha = 1 / (1 + np.exp(-10 * ((col - start_blend) / blend_width - 0.5)))
                panorama[:, col] = cv2.addWeighted(panorama[:, col], 1 - alpha, img[:, col], alpha, 0)
            return panorama

    # Bob #
    class Homography:
        def solve_homography(self, S, D):
            """
            Compute the homography matrix using Direct Linear Transformation (DLT).
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
            return H / H[2, 2]  # Normalise last element to 1


if __name__ == "__main__":
    # Saman #
    img_left = cv2.imread('s1.jpg')
    img_right = cv2.imread('s2.jpg')
    stitcher = Stitcher()
    final_result = stitcher.stitch(img_left, img_right, match_threshold=0.75, ransac_iterations=1000,
                                   ransac_threshold=5.0, blend_mode='linear')
    cv2.imshow('Final Stitched Image without Black Border', final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()