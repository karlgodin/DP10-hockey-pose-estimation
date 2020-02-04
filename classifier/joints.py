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


        perp_frames_2 = np.reshape(perp_frames, (-1, 2))

        print("test")



def main():
    parse_clip()


if __name__ == "__main__":
    main()