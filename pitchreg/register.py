import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook as tqdm


from pitchreg.sportsfield_release.utils import utils, warp, image_utils, constant_var
from pitchreg.sportsfield_release.models import end_2_end_optimization
from pitchreg.sportsfield_release.options import fake_options

opt = fake_options.FakeOptions()
opt.batch_size = 1
opt.coord_conv_template = True
opt.error_model = 'loss_surface'
opt.error_target = 'iou_whole'
opt.guess_model = 'init_guess'
opt.homo_param_method = 'deep_homography'
opt.load_weights_error_model = 'pretrained_loss_surface'
opt.load_weights_upstream = 'pretrained_init_guess'
opt.lr_optim = 1e-5
opt.need_single_image_normalization = True
opt.need_spectral_norm_error_model = True
opt.need_spectral_norm_upstream = False
opt.optim_criterion = 'l1loss'
opt.optim_iters = 150
opt.optim_method = 'stn'
opt.optim_type = 'sgd'
opt.out_dir = 'pitchreg/sportsfield_release/out/'
opt.prevent_neg = 'sigmoid'
opt.template_path = 'pitchreg/sportsfield_release/data/pitch_diagram.png'
opt.warp_dim = 8
opt.warp_type = 'homography'

constant_var.USE_CUDA = True


def reformat(frame):
    pil_image = Image.fromarray(np.uint8(frame))
    pil_image = pil_image.resize((256, 256), resample=Image.NEAREST)
    img = np.array(pil_image)
    img = utils.np_img_to_torch_img(img)
    # if opt.need_single_image_normalization:
    img = image_utils.normalize_single_image(img)
    return img


def normalise_coordinates(coordinates, frame_shape):
    frame_height, frame_width = frame_shape
    coordinates = np.array(coordinates, np.float32)
    coordinates[:, :, 0] = (coordinates[:, :, 0] / frame_width) - 0.5
    coordinates[:, :, 1] = (coordinates[:, :, 1] / frame_height) - 0.5

    return coordinates


def reverse_normalisation(transformed_coords, diagram_shape):
    diagram_height, diagram_width = diagram_shape
    reversed_coords = transformed_coords.copy()
    reversed_coords[:, 0] = (reversed_coords[:, 0] + 0.5) * diagram_width
    reversed_coords[:, 1] = (reversed_coords[:, 1] + 0.5) * diagram_height

    return reversed_coords


def calculate_pitch_coords(player_coords, homography_matrix, frame_shape, diagram_shape):
    """
    Calculates player coords from camera FOV to pitch coords using homography matrix.
    :param player_coords: List of tuples (x, y) representing player coordinates in the camera's frame.
    :param homography_matrix: Homography matrix for transformation.
    :param frame_shape: the x, y dimensions of the camera frame
    :return: List of tuples representing transformed player coordinates on the pitch.
    """

    # normalising the onscreen coords
    normie_coords = normalise_coordinates(player_coords, frame_shape)

    # Calculate inverse of homography matrix directly
    homography_matrix = homography_matrix.detach()
    homogeneous_coords = torch.from_numpy(to_homogeneous(normie_coords)).cuda()
    irl_points = []
    for n in range(homogeneous_coords.shape[0]):
        xy_warped = torch.matmul(homography_matrix, homogeneous_coords[n][0])  # H.bmm(xy)
        xy_warped, z_warped = xy_warped.split(2, dim=0)
        xy_warped = xy_warped / (z_warped + 1e-8)
        point = np.array(xy_warped.cpu().detach().numpy())
        irl_points.append(point)
    irl_points = reverse_normalisation(np.array(irl_points), diagram_shape)
    return [tuple(coord) for coord in irl_points]


def to_homogeneous(cartesian_coords):
    """
    Convert Cartesian coordinates to homogeneous coordinates.

    :param cartesian_coords: NumPy ndarray of Cartesian coordinates, shape (n, 1, 2).
    :return: Homogeneous coordinates, shape (n, 1, 3).
    """
    n = cartesian_coords.shape[0]
    ones = np.ones((n, 1, 1), dtype=cartesian_coords.dtype)
    homogeneous_coords = np.concatenate((cartesian_coords, ones), axis=2)
    return homogeneous_coords


def read_template():
    template_image = imageio.imread(opt.template_path, pilmode='RGB')
    template_image = template_image / 255.0
    if opt.coord_conv_template:
        template_image = image_utils.rgb_template_to_coord_conv_template(template_image)
    # plt.imshow(template_image)
    # plt.show()
    # covert np image to torch image, and do normalization
    template_image = utils.np_img_to_torch_img(template_image)
    if opt.need_single_image_normalization:
        template_image = image_utils.normalize_single_image(template_image)
    return template_image


def stage1(template, frame, optim_homography):

    H_inverse = torch.inverse(optim_homography)

    diagram = template / 255.0
    diagram = image_utils.rgb_template_to_coord_conv_template(diagram)
    diagram = utils.np_img_to_torch_img(diagram)

    warped_diagram = warp.warp_image(diagram, optim_homography)
    warped_diagram = utils.torch_img_to_np_img(warped_diagram)

    return warped_diagram


def warp_frame(H, diagram, frame):
    outshape = diagram.shape[1:3]
    warped_frame = warp.warp_image(frame, H, out_shape=outshape)
    plt.imshow(np.add(utils.torch_img_to_np_img(warped_frame[0]), utils.torch_img_to_np_img(diagram)))
    plt.show()
    # print("er")


def run(frame, diagram, refresh=True):
    img = reformat(frame)
    template = read_template()

    e2e = end_2_end_optimization.End2EndOptimFactory.get_end_2_end_optimization_model(opt)
    orig_homography, optim_homography = e2e.optim(img[None], template[None], refresh=refresh)

    # return stage1(diagram, optim_homography)
    return optim_homography[0]


