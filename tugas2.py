import cv2

def main():
    # fungsi panggil cam
    cap = cv2.VideoCapture(0)

    # cek web cam
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam")
        return

    while True:
        ret, frame = cap.read()

        # cek webcam
        if not ret:
            print("Error: Tidak dapat membaca frame dari webcam")
            break

        # proses grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 100)


        cv2.imshow('Frame Asli', frame)
        cv2.imshow('Deteksi Tepi (Canny)', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()