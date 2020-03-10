import json
import numpy as np

#TODO add variable frame length compatibility
n_frames = 15

def parse_clip(file_name: str):
    with open(file_name) as json_file:
        data = json.load(json_file)

        perp_frames = []
        victim_frames = []

        for frame in data:
            perp = frame["perp"]
            victim = frame["victim"]

            # TODO: add boolean to keep or remove confidence parameter
            # del perp[2::3]
            # del victim[2::3]

            perp_frames.append(perp)
            victim_frames.append(victim)

        perp_frames = get_joints(np.array(perp_frames))

        victim_frames = get_joints(np.array(victim_frames))

        return perp_frames, victim_frames


def main():
    parse_clip()
    print('done')


if __name__ == "__main__":
    main()