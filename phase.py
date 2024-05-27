from scipy import stats
import math
import numpy as np
from sklearn.cluster import KMeans

def phase_correction(past_centers, past_angle, current_OFDM):
    #print(past_phase.shape)
    corrected = current_OFDM
    corrected = corrected * np.exp(-1j * past_angle)
    centers = []
    d = []
    past_centers = np.asarray(past_centers)
    """cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    cluster_4 = []"""
    labels = []
    inti = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
    inti = np.array(inti)
    inti_complex = inti[:, 0] + 1j * inti[:, 1]
    if past_centers.shape[0] != 0:
        kmeans = KMeans(n_clusters=4, init=inti, random_state=0 ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        complex_centers = centers[:, 0] + 1j * centers[:, 1]
        distance = complex_centers - inti_complex
        d = np.array(distance)
        angle = np.angle(d)
        angle = np.mean(angle)
        if angle < 0:
            angle = 2 * np.pi + angle
        corrected = corrected * np.exp(-1j * angle)
        past_angle += angle
    else:
        kmeans = KMeans(n_clusters=4, init=inti ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        complex_centers = centers[:, 0] + 1j * centers[:, 1]

    return centers, past_angle, corrected

def gradient_correction(past_gradient, current_OFDM, known_value, positions):
    corrected = current_OFDM
    corrected = corrected * np.exp(-1j * past_gradient)
    known_angles = np.angle(known_value)
    check_angles = np.angle(corrected[positions])
    diff = np.unwrap(check_angles) - np.unwrap(known_angles)
    diff = np.unwrap(diff)
    mean_diff = np.mean(diff)
    grad, intercept, r_value, p_value, std_err = stats.linregress(positions, diff)
    if math.isclose(intercept, 2* np.pi, abs_tol=2):
        intercept = intercept - 2 * np.pi
    if math.isclose(intercept, -2* np.pi, abs_tol=2):
        intercept = intercept + 2 * np.pi

    gradient = grad * np.arange(len(current_OFDM)) + intercept

    if np.mean(np.abs(gradient)) > 1:
        gradient = 0
    corrected = corrected * np.exp(-1j * gradient)
    gradient = gradient + past_gradient
    return gradient, corrected

def combined_correction(past_centers, past_angle, past_gradient, current_OFDM):
    corrected = current_OFDM
    corrected = corrected * np.exp(-1j * past_gradient)
    corrected = corrected * np.exp(-1j * past_angle)
    centers = []
    d = []
    past_centers = np.asarray(past_centers)
    """cluster_1 = []
    cluster_2 = []
    cluster_3 = []
    cluster_4 = []"""
    labels = []
    inti = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
    inti = np.array(inti)
    inti_complex = inti[:, 0] + 1j * inti[:, 1]
    if past_centers.shape[0] != 0:
        kmeans = KMeans(n_clusters=4, init=inti, random_state=0 ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        complex_centers = centers[:, 0] + 1j * centers[:, 1]
        distance = complex_centers - inti_complex
        d = np.array(distance)
        angle = np.angle(d)
        for i in range(len(angle)):
            if angle[i] < 0:
                angle[i] = 2 * np.pi + angle[i]
        angle = np.mean(angle) - np.pi
        #print(angle)
        corrected = corrected * np.exp(-1j * angle)
        """angle = np.mean(angle)
        if angle < 0:
            angle = 2 * np.pi + angle
        corrected = corrected * np.exp(-1j * angle)"""
        past_angle += angle
    else:
        kmeans = KMeans(n_clusters=4, init=inti ).fit(np.array([np.real(corrected), np.imag(corrected)]).T)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        complex_centers = centers[:, 0] + 1j * centers[:, 1]

    predicted_ideal_angle = []
    for i in range(len(labels)):
        predicted_ideal_angle.append(np.angle(complex_centers[labels[i]]))
    predicted_ideal_angle = np.array(predicted_ideal_angle)
    predicted_ideal_angle = np.unwrap(predicted_ideal_angle)
    known_angles = np.angle(corrected)
    known_angles = np.unwrap(known_angles)
    diff = known_angles - predicted_ideal_angle
    diff = np.unwrap(diff)
    positions = np.arange(len(corrected)) + 1
    grad, intercept, r_value, p_value, std_err = stats.linregress(positions, diff)
    if math.isclose(intercept, 2* np.pi, abs_tol=2):
        intercept = intercept - 2 * np.pi
    if math.isclose(intercept, -2* np.pi, abs_tol=2):
        intercept = intercept + 2 * np.pi

    gradient = grad * np.arange(len(current_OFDM)) + intercept

    if np.mean(np.abs(gradient)) > 0.3:
        gradient = 0
    corrected = corrected * np.exp(-1j * gradient)
    gradient = gradient + past_gradient

    return centers, past_angle, gradient, corrected