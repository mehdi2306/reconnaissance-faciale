import cv2

# Chargez le classifieur de Haar
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ouvrez la webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturez une image à partir de la webcam
    ret, frame = cap.read()

    # Convertissez l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détectez les visages dans l'image
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    # Dessinez des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Affichez l'image avec les visages détectés
    cv2.imshow('Face Detection', frame)

    # Vérifiez si l'utilisateur a appuyé sur la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérez la webcam et fermez toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
