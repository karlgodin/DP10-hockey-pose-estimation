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


"""
Returns a 2D numpy array formatted according to the Interaction Recognition paper
Each row contains temporal information for one joint 

Input array:
[[ x_00, y_00, c_00, ... , x_n0, y_n0, c_n0 ],
 [ x_01, y_01, c_01, ... , x_n1, y_n1, c_n1 ],
 [                   ...                    ],
 [ x_0t, y_0t, c_0t, ... , x_nt, y_nt, c_nt ]]
 
Output array:
[[ x_00, y_00, c_00, ... , x_0t, y_0t, c_0t, 0 ],
 [ x_10, y_10, c_10, ... , x_1t, y_1t, c_1t, 1 ],
 [                   ...                       ],
 [ x_n0, y_n0, c_n0, ... , x_nt, y_nt, c_nt, 24 ]]

"""
def get_joints(player_frames : np.ndarray):
    # Obtain an array for each data point type
    player_x = player_frames[:, 0::3]
    player_y = player_frames[:, 1::3]
    player_c = player_frames[:, 2::3]

    # Create an empty array with appropriate size
    player_frames = np.empty((player_frames.shape[0] * 3, 25), dtype=player_frames.dtype)

    # Merge data points into a single 2D array
    player_frames[0::3] = player_x
    player_frames[1::3] = player_y
    player_frames[2::3] = player_c

    # Add body part information to each row and transpose the array such that each row
    # contains x_n0,y_n0,c_n0, ... x_nt, y_nt, c_nt where n is the row number and t is the frame number
    player_frames = np.append(player_frames, np.reshape(np.arange(25), (1,25)), axis=0)

    return np.transpose(player_frames)


def main():
    parse_clip()
    print('done')


if __name__ == "__main__":
    main()