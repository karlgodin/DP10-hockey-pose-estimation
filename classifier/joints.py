import json
import numpy as np

n_frames = 15

"""
class body_joint:

    def __init__(self):

"""
def parse_clip():
    with open("example_pose.json") as json_file:
        data = json.load(json_file)

        perp_frames = []
        victim_frames = []

        for frame in data:
            perp = frame["perp"]
            victim = frame["victim"]

            del perp[2::3]
            del victim[2::3]

            perp_frames.append(perp)
            victim_frames.append(victim)

        perp_frames = np.array(perp_frames)
        victim_frames = np.array(victim_frames)

        # transforming the inputs to have the joints in time for all frames

        perp_info = get_joints(perp_frames)
        victim_info = get_joints(victim_frames)

        shape_perp = perp_info.shape
        shape_victim = victim_info.shape
        print("done")

    # TODO make it an 2D array
def get_joints(player_frames: list):
    # stores the player in 1d array
    all_bodies = []
    i = 0
    j = 1

    # x = number of frames
    # y = number of points for the body (50 for the 25-body)
    x, y = player_frames.shape

    # for i+2 and j+2 so it iterates over all body parts and gets the coordinates
    while j < y:
        coordinates_of_body_part = player_frames[:, [i, j]]
        coordinates_of_body_part = coordinates_of_body_part.flatten()
        body_part_index = int(i/2)
        coordinates_of_body_part = np.append(coordinates_of_body_part, body_part_index)
        all_bodies = np.append(all_bodies, coordinates_of_body_part)

        i = i + 2
        j = j + 2
    return all_bodies


def main():
    parse_clip()


if __name__ == "__main__":
    main()