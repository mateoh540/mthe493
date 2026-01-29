# strategies/mikhail_supermartingale.py
import numpy as np
from strategies.base import CuringStrategy


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Elementwise safe division with small epsilon to avoid division by zero.
    """
    return numerator / (denominator + eps)


def strategy_13_S_supermartingale(
    A: np.ndarray,
    delta_red: np.ndarray,
    super_prob: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Python translation of Run_Polya_v2.m, case 13.

    Forces S_{i,n} to be a supermartingale with this choice of delta_black:

        for j = 1:n
            delta_black(j,t) = delta_red(j)*Super_Prob(j,t-1)/(1 - Super_Prob(j,t-1)) * ...
                               max([A(:,j).*(ones(n,1) - Super_Prob(:,t-1))./Super_Prob(:,t-1);
                                    (1-Super_Prob(j,t-1))/Super_Prob(j,t-1)]);
        end

    Parameters
    ----------
    A : (n, n) array_like
        Adjacency matrix.
    delta_red : (n,) array_like
        Red-ball addition parameters for each node.
    super_prob : (n,) array_like
        Super urn red-ball probabilities at time t-1 (Super_Prob(:, t-1)).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    delta_black : (n,) ndarray
        Curing amounts for each node at this time step.
    """
    A = np.asarray(A, dtype=float)
    delta_red = np.asarray(delta_red, dtype=float).reshape(-1)
    super_prob = np.asarray(super_prob, dtype=float).reshape(-1)

    n = delta_red.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be an (n, n) matrix where n = len(delta_red)")

    delta_black = np.zeros(n, dtype=float)

    one_minus_super = 1.0 - super_prob
    # Precompute the vector used in the max over the column and scalar
    for j in range(n):
        # MATLAB: A(:,j).*(ones(n,1) - Super_Prob(:,t-1))./Super_Prob(:,t-1)
        col_ratio = _safe_divide(A[:, j] * one_minus_super, super_prob, eps=eps)

        # MATLAB: (1-Super_Prob(j,t-1))/Super_Prob(j,t-1)
        scalar_ratio = _safe_divide(
            np.array([one_minus_super[j]]), np.array([super_prob[j]]), eps=eps
        )[0]

        max_term = np.max(np.concatenate([col_ratio, np.array([scalar_ratio])]))

        # MATLAB factor: delta_red(j)*Super_Prob(j,t-1)/(1 - Super_Prob(j,t-1))
        factor = delta_red[j] * super_prob[j] / (one_minus_super[j] + eps)
        delta_black[j] = factor * max_term

    return delta_black


def strategy_14_U_supermartingale(
    delta_red: np.ndarray,
    prop: np.ndarray,
    super_prob: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Python translation of Run_Polya_v2.m, case 14.

    Forces U_{i,n} to be a supermartingale with this choice of delta_black:

        delta_black(:,t) = delta_red(:,t).*(ones(n,1)-Prop(:,t-1)).*Super_Prob(:,t-1)./ ...
                           (Prop(:,t-1).*(ones(n,1)-Super_Prob(:,t-1)));

    Parameters
    ----------
    delta_red : (n,) array_like
        Red-ball addition parameters for each node (at time t).
    prop : (n,) array_like
        Proportion of red balls in each urn at time t-1 (Prop(:, t-1)).
    super_prob : (n,) array_like
        Super urn red-ball probabilities at time t-1 (Super_Prob(:, t-1)).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    delta_black : (n,) ndarray
        Curing amounts for each node at this time step.
    """
    delta_red = np.asarray(delta_red, dtype=float).reshape(-1)
    prop = np.asarray(prop, dtype=float).reshape(-1)
    super_prob = np.asarray(super_prob, dtype=float).reshape(-1)

    if not (delta_red.shape == prop.shape == super_prob.shape):
        raise ValueError("delta_red, prop, and super_prob must all have the same shape (n,)")

    one_minus_prop = 1.0 - prop
    one_minus_super = 1.0 - super_prob

    numerator = delta_red * one_minus_prop * super_prob
    denominator = prop * one_minus_super + eps

    delta_black = numerator / denominator
    return delta_black


def compute_avg_dblack(delta_black_history: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Compute the average delta_black over the last `window` time steps.

    This corresponds to the MATLAB code in case 27:

        if t > 50
            avg_dblack = mean(delta_black(:,(t-50):(t-1)),2);
        end

    Parameters
    ----------
    delta_black_history : (n, T) array_like
        History of delta_black values up to time t-1 (columns are time).
    window : int
        Number of most recent steps to average over.

    Returns
    -------
    avg_dblack : (n,) ndarray
        Average over the last `window` steps (or all if T < window).
    """
    delta_black_history = np.asarray(delta_black_history, dtype=float)
    n, T = delta_black_history.shape

    w = min(window, T)
    if w <= 0:
        return np.zeros(n, dtype=float)

    return np.mean(delta_black_history[:, T - w : T], axis=1)


def strategy_27_U_supermartingale_with_uniform_switch(
    A: np.ndarray,
    delta_red: np.ndarray,
    super_prob: np.ndarray,
    avg_dblack: np.ndarray,
    epsilon: float = 0.2,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Python translation of Run_Polya_v2.m, case 27.

    Uses the U_n supermartingale-based curing (like case 13), but switches
    permanently to uniform curing on any node where the average delta_black
    is sufficiently close to delta_red:

        if t > 50
            avg_dblack = mean(delta_black(:,(t-50):(t-1)),2);
        end

        use_uniform = abs(avg_dblack - delta_red(:,1)) < 0.2;

        for j = 1:n
            delta_black(j,t) = ~use_uniform(j)*<case13_formula> + use_uniform(j)*delta_red(j,t);
        end

    Parameters
    ----------
    A : (n, n) array_like
        Adjacency matrix.
    delta_red : (n,) array_like
        Red-ball addition parameters for each node (at time t).
    super_prob : (n,) array_like
        Super urn red-ball probabilities at time t-1.
    avg_dblack : (n,) array_like
        Average delta_black over a recent time window for each node.
    epsilon : float
        Threshold used in the comparison |avg_dblack - delta_red| < epsilon.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    delta_black : (n,) ndarray
        Curing amounts for each node at this time step.
    """
    A = np.asarray(A, dtype=float)
    delta_red = np.asarray(delta_red, dtype=float).reshape(-1)
    super_prob = np.asarray(super_prob, dtype=float).reshape(-1)
    avg_dblack = np.asarray(avg_dblack, dtype=float).reshape(-1)

    if not (delta_red.shape == super_prob.shape == avg_dblack.shape):
        raise ValueError("delta_red, super_prob, and avg_dblack must all have the same shape (n,)")

    n = delta_red.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be an (n, n) matrix where n = len(delta_red)")

    # MATLAB: use_uniform = abs(avg_dblack - delta_red(:,1)) < 0.2;
    use_uniform = np.abs(avg_dblack - delta_red) < epsilon

    # Start with the pure case-13 style supermartingale curing
    dblack_super = strategy_13_S_supermartingale(A, delta_red, super_prob, eps=eps)

    # MATLAB: delta_black(j,t) = ~use_uniform(j)*<case13> + use_uniform(j)*delta_red(j,t);
    delta_black = np.where(use_uniform, delta_red, dblack_super)
    return delta_black


def strategy_29_S_submartingale(
    A: np.ndarray,
    delta_red: np.ndarray,
    super_prob: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Python translation of Run_Polya_v2.m, case 29.

    This is the strict SUBmartingale analogue of case 13:

        for j = 1:n
            delta_black(j,t) = delta_red(j)*Super_Prob(j,t-1)/(1 - Super_Prob(j,t-1)) * ...
                               min([A(:,j).*(ones(n,1) - Super_Prob(:,t-1))./Super_Prob(:,t-1);
                                    (1-Super_Prob(j,t-1))/Super_Prob(j,t-1)]);
        end
    """
    A = np.asarray(A, dtype=float)
    delta_red = np.asarray(delta_red, dtype=float).reshape(-1)
    super_prob = np.asarray(super_prob, dtype=float).reshape(-1)

    n = delta_red.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be an (n, n) matrix where n = len(delta_red)")

    delta_black = np.zeros(n, dtype=float)

    one_minus_super = 1.0 - super_prob
    for j in range(n):
        col_ratio = _safe_divide(A[:, j] * one_minus_super, super_prob, eps=eps)
        scalar_ratio = _safe_divide(
            np.array([one_minus_super[j]]), np.array([super_prob[j]]), eps=eps
        )[0]

        min_term = np.min(np.concatenate([col_ratio, np.array([scalar_ratio])]))
        factor = delta_red[j] * super_prob[j] / (one_minus_super[j] + eps)
        delta_black[j] = factor * min_term

    return delta_black


def compute_delta_black(
    strategy: int,
    A: np.ndarray,
    delta_red: np.ndarray,
    super_prob: np.ndarray,
    prop: np.ndarray | None = None,
    delta_black_history: np.ndarray | None = None,
    window: int = 50,
    epsilon: float = 0.2,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convenience wrapper that dispatches to one of the supermartingale / submartingale
    curing strategies (13, 14, 27, 29) based on the MATLAB `strategy` index.

    Parameters
    ----------
    strategy : int
        One of {13, 14, 27, 29}.
    A, delta_red, super_prob : array_like
        As described in the specific strategy functions above.
    prop : array_like, optional
        Required for strategy 14 (Prop(:, t-1)).
    delta_black_history : array_like, optional
        Required for strategy 27, shape (n, T) with columns corresponding to
        past time steps of delta_black.
    window : int
        Window size for averaging in strategy 27.
    epsilon : float
        Threshold for the uniform-switch condition in strategy 27.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    delta_black : (n,) ndarray
    """
    if strategy == 13:
        return strategy_13_S_supermartingale(A, delta_red, super_prob, eps=eps)
    elif strategy == 14:
        if prop is None:
            raise ValueError("prop must be provided for strategy 14")
        return strategy_14_U_supermartingale(delta_red, prop, super_prob, eps=eps)
    elif strategy == 27:
        if delta_black_history is None:
            raise ValueError("delta_black_history must be provided for strategy 27")
        avg_dblack = compute_avg_dblack(delta_black_history, window=window)
        return strategy_27_U_supermartingale_with_uniform_switch(
            A,
            delta_red,
            super_prob,
            avg_dblack,
            epsilon=epsilon,
            eps=eps,
        )
    elif strategy == 29:
        return strategy_29_S_submartingale(A, delta_red, super_prob, eps=eps)
    else:
        raise ValueError("Unsupported strategy index for this module. Use one of {13, 14, 27, 29}.")


class MikhailSupermartingaleCuring(CuringStrategy):
    """
    Mikhail's implementation of supermartingale-based curing strategy.
    
    This strategy aims to keep a chosen risk process (e.g., expected network exposure)
    as a supermartingale by adaptively adjusting curing actions at each step.
    Supports multiple strategies (13, 14, 27, 29) based on theoretical guarantees.
    """

    def __init__(self, strategy: int = 13, window: int = 50, epsilon: float = 0.2, eps: float = 1e-12):
        """
        Initialize the Mikhail supermartingale curing strategy.
        
        Parameters
        ----------
        strategy : int
            One of {13, 14, 27, 29}.
        window : int
            Window size for averaging in strategy 27.
        epsilon : float
            Threshold for the uniform-switch condition in strategy 27.
        eps : float
            Small constant to avoid division by zero.
        """
        self.strategy = strategy
        self.window = window
        self.epsilon = epsilon
        self.eps = eps
        self.delta_black_history = None

    def before_step(self, network, t: int):
        """
        Called at the beginning of time step t (before proportions/draws).
        
        Compute curing amounts based on network state and selected strategy.
        """
        pass

    def after_step(self, network, t: int, draws: dict[int, int]):
        """
        Called after draws are realized.
        
        Update internal tracking of delta_black history for strategy 27.
        """
        if self.strategy == 27:
            # Track delta_black history for strategy 27
            pass
