import numpy as np
import pytest

from regression_demo import LinearRegressor, make_data, mse


def test_make_data_shapes():
    x, y = make_data(n=50)
    assert x.shape == (50,)
    assert y.shape == (50,)


def test_make_data_without_noise_is_exact():
    x, y = make_data(weight=3.0, bias=-2.0)
    np.testing.assert_allclose(y, 3.0 * x - 2.0)


def test_make_data_is_reproducible():
    _, y1 = make_data(noise=0.5, seed=42)
    _, y2 = make_data(noise=0.5, seed=42)
    np.testing.assert_array_equal(y1, y2)


def test_mse():
    assert mse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == pytest.approx(2.0)


def test_predict_shape():
    model = LinearRegressor()
    assert model.predict(np.zeros(8)).shape == (8,)


def test_training_reduces_loss():
    x, y = make_data()
    losses = LinearRegressor().fit(x, y, epochs=100)
    assert losses[-1] < losses[0]


def test_learns_true_line():
    x, y = make_data(weight=2.0, bias=1.0)
    model = LinearRegressor()
    model.fit(x, y, epochs=500)
    assert model.weight == pytest.approx(2.0, abs=1e-2)
    assert model.bias == pytest.approx(1.0, abs=1e-2)


def test_learns_through_noise():
    x, y = make_data(n=500, weight=-1.5, bias=0.5, noise=0.1)
    model = LinearRegressor()
    model.fit(x, y, epochs=500)
    assert model.weight == pytest.approx(-1.5, abs=0.1)
    assert model.bias == pytest.approx(0.5, abs=0.1)


def test_matches_numpy_least_squares():
    x, y = make_data(n=200, noise=0.3, seed=1)
    model = LinearRegressor()
    model.fit(x, y, epochs=2000)
    expected_w, expected_b = np.polyfit(x, y, deg=1)
    assert model.weight == pytest.approx(expected_w, abs=1e-4)
    assert model.bias == pytest.approx(expected_b, abs=1e-4)


def test_predicts_unseen_inputs():
    x, y = make_data()
    model = LinearRegressor()
    model.fit(x, y, epochs=500)
    assert model.predict(5.0) == pytest.approx(11.0, abs=0.1)
