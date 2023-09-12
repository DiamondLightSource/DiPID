from peak_detector.sslm.utils.predictor import SSLPredictor


def main():
    sdp = "peak_detector/sslm/state_dicts/40.torch"
    folder = "mm24570-1"
    nxs = 794663
    img = 20

    pred = SSLPredictor(sdp)

    pred.show_centres(folder, nxs, img)


if __name__ == "__main__":
    main()
