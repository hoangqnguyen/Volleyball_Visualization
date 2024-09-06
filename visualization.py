import glob
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch import softmax


def template_plot_tracking(template, template_size, tlwhs, obj_ids, sports, teams, scores=None, frame_id=0, fps=0.,
                           ids2=None, team_colors = [(184, 133, 27), (250, 248, 246)]):
    text_scale = 4
    text_thickness = 4
    color = (255, 0, 0)
    image = np.ascontiguousarray(np.copy(template))

    # Set colors for each team
    # team_colors = [(184, 133, 27), (250, 248, 246)]
    # heatmap = np.zeros((int(template_size[1]), int(template_size[0])), dtype=np.float32)
    # colormap = cv2.COLORMAP_JET
    # heatmap_alpha = 0.5

    # # Calculate moving averages for x and y coordinates separately
    # smoothed_x = moving_average(tlwhs[:, 0], 3)
    # smoothed_y = moving_average(tlwhs[:, 1], 3)

    team_lines = {}
    for i, tlwh in enumerate(tlwhs):
        x, y = tlwh
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        if 0 <= int(x) <= template_size[0] and 0 <= int(y) <= template_size[1]:
            if int(teams[i]) == 0:
                color = team_colors[int(teams[i])]
            elif int(teams[i]) == 1:
                color = team_colors[int(teams[i])]

            # heatmap[int(y), int(x)] += 1

            image = cv2.circle(image, (int(x), int(y)), 40, color, -1)
            (text_width, text_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)
            image = cv2.putText(image, id_text, (int(x)- text_width // 2, int(y)+ text_height // 2), cv2.FONT_HERSHEY_PLAIN, text_scale, (255,255,255),
                                thickness=text_thickness, lineType=cv2.LINE_AA)

            # Initialize the team line if it doesn't exist
            if teams[i] not in team_lines:
                team_lines[teams[i]] = []

            # Add the current player's point to the team line
            team_lines[teams[i]].append((int(x), int(y)))

            # Connect players of the same team with lines



            # for j, (cx, cy) in enumerate(tlwhs):
            #     if teams[i] == teams[j] and i != j:  # Same team and different players
            #         # if (0 <= int(x) <= template_size[0] and 0 <= int(y) <= template_size[1] and
            #         #         0 <= int(cx) <= template_size[0] and 0 <= int(cy) <= template_size[1]):
            #         cv2.line(image, (int(x), int(y)), (int(cx), int(cy)), color, 2, lineType=cv2.LINE_AA)

    # # Draw lines for each team
    # for team, line_points in team_lines.items():
    #     line_points = np.array(line_points)
    #     cv2.polylines(image, [line_points], isClosed=False, color=team_colors[team], thickness=1)

    # plt.imshow(image)
    # plt.show()

    return image


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def plot_tracking_with_template(image, template_img, tlwhs, obj_ids, court_positions, sports, teams, scores=None,
                                frame_id=0,
                                fps=0., ids2=None, team_colors = [(0, 0, 0), (245, 72, 72)]
):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    # Number of clusters (teams)
    # Set colors for each team
    # team_colors = [(123, 119, 240), (245, 72, 72)]
    team_colors = [(0, 0, 0), (245, 72, 72)]

    text_scale = 1.5
    text_thickness = 2
    line_thickness = 3

    # Calculate the size and position for the template region
    template_h, template_w = template_img.shape[:2]
    template_region_size = (im_w // 4, im_h // 4)

    template_position = (im_w - im_w // 8 - template_region_size[1], 3)

    # Resize the template image to fit the template region
    template_resized = cv2.resize(template_img, template_region_size)

    # Ensure the template region size matches the resized template image
    template_region_size = template_resized.shape[:2]

    # Check if the assignment region exceeds the dimensions of im_with_template
    if (template_position[0] + template_region_size[0] <= im_w and
            template_position[1] + template_region_size[1] <= im_h):

        # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        im_players = np.copy(im)
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            x_center = x1 + (((x1 + w) - x1) // 2)
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            color = get_color(abs(obj_id))  # Change to your desired color

            if int(teams[i]) == 0:
                color = team_colors[int(teams[i])]
            elif int(teams[i]) == 1:
                color = team_colors[int(teams[i])]

            cv2.putText(im_players, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        # (255, 255, 255),
                        color=color,
                        thickness=text_thickness, lineType=cv2.LINE_AA)
            
            cv2.circle(im_players, (int(x_center), int(y1 + h)), 5, color, -1)

            # Draw the player's direction as an ellipse
            cv2.ellipse(
                im_players,
                # center=(int(x_center), int(y1 + h)),
                center=(int(x1 + w / 2), int(y1 + h + 5)),
                axes=(int(30), int(0.35 * 30)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA
            )
            # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # blend
        alpha = .65
        im = cv2.addWeighted(im, 1 - alpha, im_players, alpha, 0)

        # for finding two closet players
        # min_distance = float('inf')
        # closest_players = None
        # # Find the two closest players
        # for i in range(len(court_positions)):
        #     for j in range(i + 1, len(court_positions)):
        #         dist = calculate_distance(court_positions[i], court_positions[j])
        #         if dist < min_distance:
        #             min_distance = dist
        #             closest_players = (i, j)

        # # Check if closest_players is not None before subscripting
        # if closest_players is not None:
        #     # print("closest players : ", closest_players)
        #     player_0 = tlwhs[closest_players[0]]
        #     player_0_center = player_0[0] + (((player_0[0] + player_0[2]) - player_0[0]) // 2)
        #     # print("closest player_0 : ", player_0)
        #     player_1 = tlwhs[closest_players[1]]
        #     player_1_center = player_1[0] + (((player_1[0] + player_1[2]) - player_1[0]) // 2)
        #     # print("closest player_1 : ", player_1)

        #     # Draw a line between the two closest players
        #     cv2.line(im, (int(player_0_center), int(player_0[1] + player_0[3])), (int(player_1_center),
        #                                                                           int(player_1[1] + player_1[3])),
        #              (80, 200, 120), 2)
        # else:
        #     pass
            # Handle the case where no players are found
            # print("No players found.")

        # Copy the original image to avoid modifying it
        im_with_template = np.copy(im)

        # Paste the resized template image onto the top-right corner
        im_with_template[template_position[1]:template_position[1] + template_region_size[0],
        template_position[0]:template_position[0] + template_region_size[1]] = template_resized

        return cv2.addWeighted(im, 0.1, im_with_template, 0.9, 0)
    else:
        # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            x_center = x1 + (((x1 + w) - x1) // 2)
            # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            color = get_color(abs(obj_id))

            if int(teams[i]) == 0:
                color = team_colors[int(teams[i])]
            elif int(teams[i]) == 1:
                color = team_colors[int(teams[i])]

            # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.circle(im, (int(x_center), int(y1 + h)), 8, color, -1)
            cv2.putText(im, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (255, 255, 255),
                        thickness=text_thickness)
            cv2.ellipse(
                im,
                # center=(int(x_center), int(y1 + h)),
                center=(int(x1 + w / 2), int(y1 + h)),
                axes=(int(70), int(0.35 * 70)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=3,
                lineType=cv2.LINE_4
            )

        return im


def plot_tracking(image, tlwhs, obj_ids, teams, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    # Set colors for each team
    team_colors = [(123, 119, 240), (245, 72, 72)]
    # Transparency value

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        x_center = x1 + (((x1 + w) - x1) // 2)
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        if int(teams[i]) == 0:
            color = team_colors[int(teams[i])]
        elif int(teams[i]) == 1:
            color = team_colors[int(teams[i])]

        # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # cv2.circle(im, (int(x_center), int(y1 + h)), 8, color, -1)
        cv2.putText(im, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                    color=color,
                    thickness=text_thickness, lineType=cv2.LINE_AA)

        cv2.ellipse(
            im,
            # center=(int(x_center), int(y1 + h)),
            center=(int(x1 + w / 2), int(y1 + h)),
            axes=(int(70), int(0.35 * 70)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,
            lineType=cv2.LINE_4
        )

    # plt.imshow(im)
    # plt.show()

    return im

def custom_filter(size=5, gamma=1):
    # gamma: increase gamma to put more weight to the current frame
    raw_value = np.power(np.arange(1, size + 1), gamma)
    raw_value[size // 2 + 1:] = raw_value[: size // 2][::-1]
    return softmax(raw_value)


def load_and_smooth_homography(
        homography_folder, frames_list, window_size=5, gamma=1.5
):
    homography_matrix_list = []
    for frame_idx in frames_list:
        search = f"{homography_folder}/{frame_idx:05d}.npy"
        homography_file = glob.glob(search)[0]
        homography_matrix = np.load(homography_file)
        homography_matrix_list.append(
            {"frame": frame_idx, "matrix": homography_matrix.reshape(1, 9)}
        )

    homography_matrix_df = pd.DataFrame(homography_matrix_list)

    # smooth
    stacked_homography_matrix = np.concatenate(
        homography_matrix_df["matrix"].tolist(), axis=0
    )

    # we'll not use zero-padding which makes things jittering at the start, instead we pad using the homography from
    # the first frame
    start_paddings = [stacked_homography_matrix[[0]]] * (window_size // 2)
    end_paddings = [stacked_homography_matrix[[-1]]] * (window_size // 2)

    stacked_homography_matrix = np.concatenate(
        (*start_paddings, stacked_homography_matrix, *end_paddings), axis=0
    )

    # kernel = softmax(normal_filter(window_size))
    kernel = custom_filter(window_size, gamma=gamma)

    for idx in range(stacked_homography_matrix.shape[-1]):
        stacked_homography_matrix[
        window_size // 2: -window_size // 2 + 1, idx
        ] = np.convolve(stacked_homography_matrix[:, idx], kernel, mode="valid")

    stacked_homography_matrix = stacked_homography_matrix.reshape(-1, 3, 3)
    stacked_homography_matrix = stacked_homography_matrix[
                                window_size // 2: -window_size // 2 + 1
                                ]
    return stacked_homography_matrix

def calculate_minimap_position(homography_folder, frame_df, window_size=5, gamma=1.5):
    frames_list = sorted(list(set(frame_df.frame)))
    frame_df["x"] = frame_df["x"] + frame_df["w"] / 2
    frame_df["y"] = frame_df["y"] + frame_df["h"]
    frame_df["z"] = 1.0
    stacked_homography_matrix = load_and_smooth_homography(
        homography_folder, frames_list, window_size=window_size, gamma=gamma
    )
    for frame_idx in frames_list:
        # 1. homography projection
        homography_matrix = stacked_homography_matrix[frame_idx]
        player_position_in_frame = frame_df.loc[
            frame_df["frame"] == frame_idx, ["x", "y", "z"]
        ].to_numpy()
        player_position_in_minimap = (homography_matrix @ player_position_in_frame.T).T
        player_position_in_minimap /= player_position_in_minimap[:, [-1]]
        player_position_in_minimap = player_position_in_minimap.astype(int)
        # 2. replace x y z to visualize
        frame_df.loc[
            frame_df["frame"] == frame_idx, ["x", "y", "z"]
        ] = player_position_in_minimap

    return frame_df


def smooth_positions(frame_df, window_size=3, gamma=0.85):
    kernel = custom_filter(window_size, gamma)
    players = frame_df.player.unique().tolist()
    for player_idx in players:
        data = frame_df.loc[frame_df.player == player_idx, ["x", "y"]].to_numpy()
        if data.shape[0] > window_size:
            smooth_xy = np.convolve(data[:, 0], kernel, mode="same"), np.convolve(
                data[:, 1], kernel, mode="same"
            )
            smooth_xy = np.array(smooth_xy, dtype=int).T
            frame_df.loc[frame_df.player == player_idx, ["x", "y"]] = smooth_xy
    return frame_df


def visualize_court_demo(vis_folder, path, tracking_file_path, court_position_path, sports):
    result_folder = osp.join(vis_folder, f"result_{Path(path).stem}")
    os.makedirs(result_folder, exist_ok=True)
    template_image_path = " "
    if sports == "volleyball":
        template_image_path = './template/volleyball_color.png'
        template_size = (1280, 720)
    template = cv2.imread(template_image_path)
    # threshold template to create bw
    _, template_bw = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

    # read tracking result text file
    frame_df = pd.read_csv(tracking_file_path, names=["frame", "track_id", "x", "y", "w", "h", "score", "a",
                                                      "b", "c", "team"])

    court_df = pd.read_csv(court_position_path, names=["frame", "track_id", "x", "y", "team"])
    frames_list = sorted(list(set(frame_df.frame)))

    # smooth out process
    # court_frame_df = calculate_minimap_position(img_folder, court_frame_df, window_size=3, gamma=1.5)
    # court_frame_df = smooth_positions(court_frame_df, window_size=3, gamma=0.85)

    for frame_idx in frames_list:
        # read data from players position on image
        frame_info = frame_df[frame_df.frame == frame_idx]
        court_frame_info = court_df[court_df.frame == frame_idx]

        # read image according to frame id
        img_name = osp.join(path, "{}.jpg".format(str(frame_idx).zfill(6)))
        frame = cv2.imread(img_name)
        tlwh = frame_info[["x", "y", "w", "h"]].to_numpy()
        obj_ids = frame_info["track_id"].to_numpy()
        teams = frame_info["team"].to_numpy()
        court_position = court_frame_info[["x", "y"]].to_numpy()

        # read homography for check
        npy_name = osp.join(path, "{}.npy".format(str(frame_idx).zfill(6)))

        homography = np.load(npy_name)
        is_identity = np.array_equal(homography, np.eye(3))

        # overlay on original image
        init_H_inv = np.linalg.inv(homography)

        im_out = cv2.warpPerspective(template_bw, init_H_inv, frame.shape[:2][::-1])

        # show initial guess overlay
        show_image = np.copy(frame)
        valid_index = im_out[:, :, 0] > 0.0
        overlay = (frame[valid_index].astype('float32') + im_out[valid_index].astype(
            'float32')) / 2
        show_image[valid_index] = overlay
        team_colors = [(0, 0, 0), (245, 72, 72)]

        if not is_identity:
            template_img = template_plot_tracking(template, template_size, court_position,
                                                  obj_ids, sports, teams, frame_id=frame_idx, team_colors=team_colors)
            online_im = plot_tracking_with_template(
                show_image, template_img, tlwh, obj_ids, court_position, sports, teams,
                frame_id=frame_idx, team_colors=team_colors)
        else:
            online_im = plot_tracking(show_image, tlwh, obj_ids, teams, frame_id= frame_idx)

        cv2.imwrite(osp.join(result_folder, "{}_result.jpg".format(str(frame_idx).zfill(6))), online_im)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    root_directory = os.path.abspath("videos")
    template_image_path = os.path.abspath("template/volleyball_binary.png")
    sports = "volleyball"
    for vdo_name in sorted(os.listdir(root_directory)):
        vdo_path = os.path.join(root_directory, vdo_name)
        if os.path.exists(vdo_path) and os.path.isdir(vdo_path):
            tracking_file = osp.join(root_directory, Path(vdo_path).stem + ".txt")
            court_file = osp.join(root_directory, "court_" + Path(vdo_path).stem + ".txt")
            print(f"\n -------------- Visualizing tracking result for {vdo_path} -------------- ")
            visualize_court_demo(root_directory, vdo_path, tracking_file, court_file, sports)
            exit()