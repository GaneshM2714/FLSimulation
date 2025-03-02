# server.py
import flwr as fl
from flwr.server.strategy import (
    FedAvg,
    FedProx,
    FedAdagrad,
    FedAdam,
    Strategy,
)
from typing import List, Tuple, Optional, Dict
from flwr.server import Server, ServerConfig
from flwr.common import Parameters, MetricsAggregationFn, EvaluateIns, EvaluateRes, FitIns, FitRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.state import State
from flwr.server.app import run_server
from flwr.server.strategy import FedAvgMAB

import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Define the list of strategies to be used in sequence with parameters
STRATEGIES_SEQUENCE = [
    FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=4,
    ),
    FedProx(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=4,
        proximal_mu=0.1,
    ),
    FedAdagrad(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=4,
        learning_rate_initial=0.01,
        l2_regularization=0.001,
    ),
    FedAdam(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=4,
        learning_rate_initial=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        l2_regularization=0.0001,
    ),
]

# Number of rounds to use each strategy for
ROUNDS_PER_STRATEGY = 2

class StrategyCycler(Strategy):
    """
    A strategy that cycles through a list of strategies, using each for a
    predefined number of rounds, and stores accuracy.
    """

    def __init__(self, strategies: List[Strategy], rounds_per_strategy: int):
        super().__init__()
        self.strategies = strategies
        self.rounds_per_strategy = rounds_per_strategy
        self.current_strategy_index = 0
        self.round_count_for_strategy = 0
        self.current_strategy = self.strategies[self.current_strategy_index]
        self.accuracy_history = []  # List to store accuracy per round and strategy

    def __repr__(self) -> str:
        return "StrategyCycler"

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.current_strategy.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[str, FitIns]]:
        """Configure the next round of training."""
        self._update_strategy_if_needed(server_round)
        return self.current_strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[str, FitRes]],
        failures: List[str],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        return self.current_strategy.aggregate_fit(server_round, results, failures)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[str, EvaluateIns]]:
        """Configure the next round of evaluation."""
        self._update_strategy_if_needed(server_round)
        return self.current_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[str, EvaluateRes]],
        failures: List[str],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and store accuracy."""
        aggregation_result = self.current_strategy.aggregate_evaluate(server_round, results, failures)
        if aggregation_result is not None:
            loss, metrics = aggregation_result
            if metrics and "accuracy" in metrics:
                accuracy = metrics["accuracy"]
                self.accuracy_history.append({
                    "round": server_round,
                    "strategy": type(self.current_strategy).__name__,
                    "accuracy": accuracy,
                })
                print(f"Round {server_round}, Strategy: {type(self.current_strategy).__name__}, Accuracy: {accuracy:.4f}")
        return aggregation_result

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar]]]:
        """Evaluate global model (server-side evaluation)."""
        return self.current_strategy.evaluate(server_round, parameters)

    def configure_initial_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Configure initial model parameters."""
        return self.current_strategy.configure_initial_parameters(client_manager)

    def create_server_state(self, client_manager: ClientManager) -> State:
        """Create initial server state."""
        return self.current_strategy.create_server_state(client_manager)

    def _update_strategy_if_needed(self, server_round: int):
        """Update the strategy if the round count for the current strategy is reached."""
        if (server_round - 1) % self.rounds_per_strategy == 0 and server_round > 1:
            self.current_strategy_index = (self.current_strategy_index + 1) % len(self.strategies)
            self.current_strategy = self.strategies[self.current_strategy_index]
            print(f"\n---------- Switching to Strategy: {type(self.current_strategy).__name__} ----------\n")


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregation function for weighted average metrics."""
    totals = sum([num_examples for num_examples, _ in metrics])
    if totals == 0:
        return {}
    weighted_metrics: Dict[str, Scalar] = {}
    for num_examples, metric in metrics:
        for key, val in metric.items():
            weighted_metrics[key] = weighted_metrics.get(key, 0.0) + val * num_examples
    return {key: val / totals for key, val in weighted_metrics.items()}


def plot_accuracy(accuracy_history: List[Dict]):
    """Plotting function to visualize accuracy vs rounds for each strategy."""
    strategy_names = set(item['strategy'] for item in accuracy_history)
    plt.figure(figsize=(10, 6))
    for strategy_name in strategy_names:
        strategy_data = [item for item in accuracy_history if item['strategy'] == strategy_name]
        rounds = [item['round'] for item in strategy_data]
        accuracies = [item['accuracy'] for item in strategy_data]
        plt.plot(rounds, accuracies, label=strategy_name)

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Accuracy vs Rounds per Strategy")
    plt.legend()
    plt.grid(True)
    plt.savefig("fl_accuracy_plot.png")  # Save plot to a file
    plt.show()


def main():
    """Start the FL Server with strategy cycling and plot accuracy."""
    strategy_cycler = StrategyCycler(STRATEGIES_SEQUENCE, ROUNDS_PER_STRATEGY)

    server_config = ServerConfig(num_rounds=len(STRATEGIES_SEQUENCE) * ROUNDS_PER_STRATEGY)

    history = fl.server.start_server(  # Capture the History object (not directly used here but could be)
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy_cycler,
        # metrics_aggregation_fn=weighted_average, # Uncomment if you are using custom metrics
    )

    plot_accuracy(strategy_cycler.accuracy_history) # Plot accuracy after server finishes

if __name__ == "__main__":
    main()