import numpy as np
import matplotlib.pyplot as plt



def eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    for i in range(len(eigenvalues)):
        lambda_v = eigenvalues[i] * eigenvectors[:, i]
        Av = np.dot(matrix, eigenvectors[:, i])
        print(f"Eigenvalue {i + 1}: {eigenvalues[i]}")
        print(f"Eigenvector {i + 1}: {eigenvectors[:, i]}")
        print(f"A * v = {Av}")
        print(f"λ * v = {lambda_v}")
        print(f"Check: {np.allclose(Av, lambda_v)}\n")
    return eigenvalues, eigenvectors



def load_and_process_image(image_path):
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.title('Original Image')
    plt.show()

    image_bw = np.mean(image, axis=2)
    plt.imshow(image_bw, cmap='gray')
    plt.title('Black and White Image')
    plt.show()

    return image_bw


def pca_image_compression(image, variance_threshold=0.95):
    image_centered = image - np.mean(image, axis=0)
    covariance_matrix = np.cov(image_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    principal_components = eigenvectors[:, :num_components]
    transformed_data = np.dot(image_centered, principal_components)
    reconstructed_image = np.dot(transformed_data, principal_components.T) + np.mean(image, axis=0)

    return reconstructed_image, cumulative_variance, num_components


def perform_pca(image_bw, variance_threshold=0.95):
    reconstructed_image, cumulative_variance, num_components = pca_image_compression(image_bw, variance_threshold)

    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Reconstructed Image with {num_components} Components')
    plt.show()

    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.title('Cumulative Variance Explained by PCA Components')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.show()

    return reconstructed_image, cumulative_variance, num_components


def reconstruct_image_with_different_components(image_bw, num_components_list):
    image_centered = image_bw - np.mean(image_bw, axis=0)
    covariance_matrix = np.cov(image_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    num_components_list.sort()
    n = len(num_components_list)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    for idx, num_components in enumerate(num_components_list):
        principal_components = eigenvectors[:, :num_components]
        transformed_data = np.dot(image_centered, principal_components)
        reconstructed_image = np.dot(transformed_data, principal_components.T) + np.mean(image_bw, axis=0)

        ax = axes[idx // cols, idx % cols]
        ax.imshow(reconstructed_image, cmap='gray')
        ax.set_title(f'{num_components} Components')

    for i in range(n, rows * cols):
        fig.delaxes(axes[i // cols, i % cols])

    plt.tight_layout()
    plt.show()



def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    inverse_diagonalized_key_matrix = np.linalg.inv(diagonalized_key_matrix)
    decrypted_vector = np.dot(inverse_diagonalized_key_matrix, encrypted_vector)
    decrypted_message = ''.join([chr(int(round(num))) for num in decrypted_vector])
    return decrypted_message


def main():

    matrix = np.array([[4, -1], [6, -2]])
    eigenvalues, eigenvectors = eigenvalues_and_eigenvectors(matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)


    image_path = '/Users/arturnanivskij/PycharmProjects/lab2/IMG_7677-Enhanced-NR-3.jpg'
    image_bw = load_and_process_image(image_path)
    reconstructed_image, cumulative_variance, num_components = perform_pca(image_bw, variance_threshold=0.95)
    num_components_list = [10, 20, 30, 50, 80, 100, 250, 340, num_components]
    reconstruct_image_with_different_components(image_bw, num_components_list)


    original_message = "Hello, World!"
    key_matrix = np.random.rand(13, 13)
    print("Original Message:", original_message)
    encrypted_message = encrypt_message(original_message, key_matrix)
    print("Encrypted Message:", encrypted_message)
    decrypted_message = decrypt_message(encrypted_message, key_matrix)
    print("Decrypted Message:", decrypted_message)


if __name__ == "__main__":
    main()
