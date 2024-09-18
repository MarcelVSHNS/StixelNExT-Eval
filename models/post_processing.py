import pandas as pd
import stixel as stx
import torch
from einops import rearrange


def revert_class(prediction: torch.Tensor,
                 anchors: pd.DataFrame,
                 calib: stx.stixel_world_pb2.CameraInfo,
                 prob: float = 0.9,
                 u_scale: int = 8,
                 d_scale: float = 50.0,
                 img_height: int = 1280,
                 four_attr: bool = False
                 ) -> stx.StixelWorld:
    """ extract stixel information from prediction """
    d_scale = d_scale * 0.1
    pred_np = prediction.numpy()
    stxl_wrld = stx.StixelWorld()
    stxl_wrld.context.calibration = calib
    columns = rearrange(pred_np, "a n u -> u n a")
    for u in range(len(columns)):
        # print(f"Col1: {column.shape}")
        for n in range(len(columns[u])):
            # print(f"candidate1: {candidate.shape}")
            if four_attr:
                if columns[u][n][3] >= prob:
                    stxl = stx.Stixel()
                    stxl.u = int(u * u_scale)
                    stxl.vT = int(columns[u][n][1] * img_height)
                    stxl.vB = int(columns[u][n][0] * img_height)
                    stxl.d = columns[u][n][2] * d_scale + anchors[f'{u}'][n]
                    stxl.confidence = columns[u][n][3]
                    stxl.width = u_scale
                    stxl_wrld.stixel.append(stxl)
            else:
                if columns[u][n][2] >= prob:
                    stxl = stx.Stixel()
                    stxl.u = int(u * u_scale)
                    stxl.vT = int(columns[u][n][1] * img_height + 1)
                    stxl.vB = int(columns[u][n][0] * img_height + 1)
                    stxl.d = anchors[f'{u}'][n]
                    stxl.confidence = columns[u][n][2]
                    stxl.width = u_scale
                    stxl_wrld.stixel.append(stxl)
    return stxl_wrld


def revert_segm(prediction: torch.Tensor,
                anchors: pd.DataFrame,
                calib: stx.stixel_world_pb2.CameraInfo,
                prob: float = 0.9,
                u_scale: int = 8,
                v_scale: int = 8
                ) -> stx.StixelWorld:
    pred_np = prediction.numpy()
    stxl_wrld = stx.StixelWorld()
    stxl_wrld.context.calibration.K.extend(calib.K)
    stxl_wrld.context.calibration.T.extend(calib.T)
    columns = rearrange(pred_np, "d h w -> w d h")
    for u in range(len(columns)):
        for d in range(len(columns[u])):
            stixel_start = 0
            in_stixel = False
            stixel_prob = []
            for v in range(len(columns[u][d])):
                if in_stixel:
                    if columns[u][d][v] < prob:
                        stxl = stx.Stixel()
                        stxl.u = int(u * u_scale)
                        stxl.vT = int(stixel_start * v_scale)
                        stxl.vB = int(v * v_scale)
                        stxl.d = anchors[f'{u}'][d]
                        stxl.confidence = sum(stixel_prob) / len(stixel_prob)
                        stxl.width = u_scale
                        stxl_wrld.stixel.append(stxl)
                        in_stixel = False
                    else:
                        stixel_prob.append(columns[u][d][v])
                else:
                    if columns[u][d][v] >= prob:
                        stixel_start = v
                        stixel_prob.append(columns[u][d][v])
                        in_stixel = True
                    else:
                        pass
    # ground truth injection to evaluate transformation etc.
    if False:
        gt: stx.StixelWorld = stx.read('samples/191862526745161106_1400_000_1420_000_25_FRONT.stx1')
        del stxl_wrld.stixel[:]
        stxl_wrld.stixel.extend(gt.stixel)
    return stxl_wrld
