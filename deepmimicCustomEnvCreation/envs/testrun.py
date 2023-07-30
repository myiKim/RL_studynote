import gym 
from gym.envs.registration import register

import numpy as np
from array2gif import write_gif
import cv2

register(
    id='hmyi-v0',
    entry_point='deep_mimic_env:hmyifEnv'
)

print("All ENVS :", [env_spec for env_spec in gym.envs.registration.registry])

def print_env_info(env_obj):
    
    print("===================== INFO (before running) ==================")
    print("Action Space : ", env_obj.action_space)
    # print("Internal env : ", env_obj._internal_env)
    # print("Num Agents : ", env_obj._internal_env.get_num_agents())
    print("==============================================================")
    return True

def initialize_env():
    global env
    env = gym.make("hmyi-v0")
    SEED =77777
    env.action_space.seed(SEED)
    env.reset()
    
    return print_env_info(env)

def drive_main(range_):

    recordings = []

    for tt in range(range_):
        
        observation, reward, terminated, info = env.step(env.action_space.sample())
        rgb_array = env.render("rgb_array")
        recordings.append(rgb_array)
        print("At frame == ", tt, ", writing RGB array with size: ", rgb_array.shape)
        if tt % 100 ==0:
            print("[Observation statistics] we observe : ", observation, "with reward : ", reward)
        
        if terminated: 
            env.reset()
            
    return {"finishedAt": tt, "recordings": recordings}

def write_to_gif(gifpath, simulation, fps = 14):
    recordings = []
    print("trp process started .. ")
    for frame_index, frame in enumerate(simulation["recordings"]):
        transposed = np.transpose(frame, (1,0,2))
        print("shape == ", transposed.shape)
        frame_with_text = transposed.copy()
        text = f"Frame: {frame_index}"  # Text to display (Frame: 0, Frame: 1, etc.)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 0, 0)  # White color (BGR format)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = transposed.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(frame_with_text, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        recordings.append(frame_with_text)
    print("trp process finished .. ")
    print("Now, writing gif .. ")

    #Use write_gif pkg to write gif
    write_gif(recordings, gifpath, fps=fps)

    return True

# def write_to_mp4(mp4path, simulation, fps = 14):
#     recordings = []
#     transposed = None
#     print("trp process started .. ")
#     for frame in simulation["recordings"]:
#         transposed = np.transpose(frame, (1,0,2))
#         print("shape == ", transposed.shape)
        
#         recordings.append(transposed.astype(np.uint8))
#     print("trp process finished .. ")
#     print("Now, writing mp4 .. ")
#     (fW, fH, _) = transposed.shape
#     #Use opencv pkg to write mp4
#     codec = cv2.VideoWriter_fourcc(*'H264')
#     out = cv2.VideoWriter(mp4path, codec, fps, (fW,fH))
#     for frame in recordings:
#         bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         out.write(bgr_frame)
#     out.release()   

#     return True
    
def main():
    
    print(initialize_env())
    NUM_STEPS = 550
    simulation = drive_main(NUM_STEPS)
    # write_to_mp4('../result_files/debug_video.mp4', simulation, 14) 
    return write_to_gif('../result_files/debug_video.gif', simulation, 14)    


if __name__ == '__main__':
    main()
    