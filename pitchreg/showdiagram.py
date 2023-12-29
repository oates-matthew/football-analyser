import cv2
import numpy as np


def represent_players(diagram, coordinates, player_ids, player_teams, circle_radius=20, colour0=(30, 10, 240), colour1=(255, 100, 10)):
    diagram_with_circles = diagram.copy()
    for coord, id in zip(coordinates, player_ids):
        team = player_teams[id]
        coord = tuple(tuple(map(int, coord)) for c in coord)[0]
        # coord = coord[::-1]
        colour = colour0 if team == 1 else colour1
        diagram_with_circles = cv2.circle(diagram_with_circles, coord, circle_radius, colour, thickness=-1)

    return diagram_with_circles


def overwrite(frame, diagram, alpha=0.7):
    """
    Overwrite the frame with a transparent diagram.

    :param frame: The original video frame.
    :param diagram: The diagram to be overlaid.
    :param alpha: The transparency level of the diagram, between 0 (fully transparent) and 1 (fully opaque).
    :return: The frame with the overlaid diagram.
    """
    # Resize the diagram
    diagram_resized = cv2.resize(diagram, (frame.shape[1] // 4, frame.shape[0] // 4))

    # Calculate the position for overlay
    x_offset = frame.shape[1] // 2 - diagram_resized.shape[1] // 2
    y_offset = frame.shape[0] - diagram_resized.shape[0]

    # Extract the region of interest
    roi = frame[y_offset:y_offset + diagram_resized.shape[0], x_offset:x_offset + diagram_resized.shape[1]]

    # Create a mask for the diagram and its inverse mask
    mask = cv2.inRange(diagram_resized, (1, 1, 1), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of diagram in ROI
    # roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of diagram from diagram image.
    diagram_fg = cv2.bitwise_and(diagram_resized, diagram_resized, mask=mask)

    # Blend the diagram and the region of interest
    blended = cv2.addWeighted(diagram_fg, alpha, roi, 1 - alpha, 0)

    # Replace the region in the original frame with the blended result
    frame[y_offset:y_offset + diagram_resized.shape[0], x_offset:x_offset + diagram_resized.shape[1]] = blended

    return frame