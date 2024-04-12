from PIL import Image
import numpy as np

class Gradient:
    """
    Parent class to define repeateable methods and variabls. Takes in the following parameters:
        width: width in pixels
        height: height in pixels
        heatmap: a map of RGB points and their corresponding distance from the previous one
                    example_heatmap = [
                        [0.0, (0, 0, 0)],
                        [0.5, (0.75, 0.25, 0.55)],
                        [1.0, (1.0, 1.0, 1.0)]
                    ]
        path: output image path

    All gradients can be generated using the same method name - generate_gradient()
    """

    def __init__(self, width, height, heatmap, path) -> None:
        self.width = width
        self.height = height
        self.heatmap = heatmap
        self.path = path

    def save(self):
        self.image.save(self.path)

    def interpolate_colors(self, normalized_distances):
        gradient = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for i in range(3):  # Loop over RGB channels
            for j in range(len(self.heatmap) - 1):  # Interpolate between heatmap points
                start_color = np.array(self.heatmap[j][1])
                end_color = np.array(self.heatmap[j + 1][1])
                t = (normalized_distances - heatmap[j][0]) / (self.heatmap[j + 1][0] - self.heatmap[j][0])
                mask = (self.heatmap[j][0] <= normalized_distances) & (normalized_distances <= self.heatmap[j + 1][0])
                gradient[mask, i] = (1 - t[mask]) * start_color[i] + t[mask] * end_color[i]
        return gradient

class LinearGradient(Gradient):
    """
    Class creating linear gradients. Takes in the following arguments:
        width: width in pixels
        height: height in pixels
        heatmap: a map of RGB points and their corresponding distance from the previous one
                    example_heatmap = [
                        [0.0, (0, 0, 0)],
                        [0.5, (0.75, 0.25, 0.55)],
                        [1.0, (1.0, 1.0, 1.0)]
                    ]
        path: output image path
        ---
        angle: angle by which the gradient is rotated. Takes in int value in %.
    """
    def __init__(self, width, height, heatmap, path, angle) -> None:
        super().__init__(width, height, heatmap, path)
        self.angle = angle

    def generate_gradient(self):
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        angle_radians = np.deg2rad(self.angle)
        extended_width = int(np.ceil(self.width * self._find_extension_factor()))
        distances = x * np.cos(angle_radians) + y * np.sin(angle_radians)
        normalized_distances = distances / extended_width
        gradient = self.interpolate_colors(normalized_distances)
        
        gradient_image = Image.fromarray(gradient)
        gradient_image.save(self.path)

    def _find_extension_factor(self):
        if 0 < self.angle >= 90:
            factor = 1.0 - 0.1 * (self.angle / 90)  # Bigger factor for angles closer to 90 degrees
        else:
            factor = 1.0 + 0.1 * ((180 - self.angle) / 90)  # Smaller factor for angles closer to 0 or 180 degrees
        return factor

class RadialGradient(Gradient):
    """
    Class creating Radial Gradients. Takes in the following arguments:
        width: width in pixels
        height: height in pixels
        heatmap: a map of RGB points and their corresponding distance from the previous one
                    example_heatmap = [
                        [0.0, (0, 0, 0)],
                        [0.5, (0.75, 0.25, 0.55)],
                        [1.0, (1.0, 1.0, 1.0)]
                    ]
        path: output image path
    """
    def __init__(self, width, height, heatmap, path) -> None:
        super().__init__(width, height, heatmap, path)

    
    def generate_gradient(self):
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)

        x_center = self.width / 2
        y_center = self.height / 2
        distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        normalized_distance = distance / (np.sqrt((self.width/2)**2 + (self.height/2)**2))
        gradient = self.interpolate_colors(normalized_distance)

        gradient_image = Image.fromarray(gradient)
        gradient_image.save(self.path)


class EllipticalGradient(Gradient):
    """
    Class creating Elliptical Gradients. Takes in the following arguments:
    width: width in pixels
        height: height in pixels
        heatmap: a map of RGB points and their corresponding distance from the previous one
                    example_heatmap = [
                        [0.0, (0, 0, 0)],
                        [0.5, (0.75, 0.25, 0.55)],
                        [1.0, (1.0, 1.0, 1.0)]
                    ]
        path: output image path
        ---
        major_axis: defines longer diameter of the ellipse
        minor_axis: defines shorter diameter of the ellipse
    """
    def __init__(self, width, height, heatmap, path, major_axis, minor_axis) -> None:
        super().__init__(width, height, heatmap, path)
        self.major_axis = major_axis
        self.minor_axis = minor_axis


    def generate_gradient(self):
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)

        x_center = self.width / 2
        y_center = self.height / 2

        ellipse_width = self.width * 1.5
        ellipse_height = self.height * 1.5

        distance_major = np.abs(x - x_center)
        distance_minor = np.abs(y - y_center)

        normalized_distance_major = distance_major / (ellipse_width / 2)
        normalized_distance_minor = distance_minor / (ellipse_height / 2)
        normalized_distance = np.sqrt(normalized_distance_major**2 + normalized_distance_minor**2)
        gradient = self.interpolate_colors(normalized_distance)

        gradient_image = Image.fromarray(gradient)
        gradient_image.save(self.path)

class CenterPointGradient(Gradient):
    """
    Class creating CenterPoint Gradients (gradients rotating around one specified point). Takes in the following arguments:
    width: width in pixels
        height: height in pixels
        heatmap: a map of RGB points and their corresponding distance from the previous one
                    example_heatmap = [
                        [0.0, (0, 0, 0)],
                        [0.5, (0.75, 0.25, 0.55)],
                        [1.0, (1.0, 1.0, 1.0)]
                    ]
        path: output image path
        ---
        center_x: x value to center the gradient around 
        center_y: y value to center the gradient around
        angle: angle in degrees by which the gradient will be rotated
    """
    def __init__(self, width, height, heatmap, path, center_x, center_y, angle) -> None:
        super().__init__(width, height, heatmap, path)
        self.center_x = center_x
        self.center_y = center_y
        self.angle = np.deg2rad(float(angle))

    def generate_gradient(self):
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)

        relative_x = x - self.center_x
        relative_y = y - self.center_y
        theta = np.arctan2(relative_y, relative_x)

        self.angle = self.angle % (2 * np.pi)
        angular_distance = np.abs((theta - self.angle + np.pi) % (2 * np.pi) - np.pi) / np.pi
        gradient = self.interpolate_colors(angular_distance)

        gradient_image = Image.fromarray(gradient)
        gradient_image.save(self.path)

"""
EXAMPLE USAGE:
"""

heatmap = [
    [0.0, (255, 0, 0)],    # Red at the beginning
    [0.5, (0, 255, 0)],    # Green in the middle
    [1.0, (0, 0, 255)]     # Blue at the end
]

linear_gradient = LinearGradient(500, 250, heatmap, 'gradient_linear.png', 20).generate_gradient()
radial_gradient = RadialGradient(500, 250, heatmap, 'gradient_radial.png').generate_gradient()
elliptic_gradient = EllipticalGradient(500, 250, heatmap, 'gradient_elliptical.png', 300, 25).generate_gradient()
centered_gradient = CenterPointGradient(500, 250, heatmap, 'gradient_point_centered.png', 50, 160, 45).generate_gradient()


