import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg

LOG_DIR = '/content/drive/MyDrive/MajorProject/logs/experiment2/federated/'

class FederatedStrategy(FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result =  super().aggregate_evaluate(rnd, results, failures)
        loss_aggregated, metrics_aggregated = aggregated_result

        # Log evaluation loss and accuracy
        with open(LOG_DIR + 'server_evaluation.log', 'a') as f:
            f.write(f'{rnd},{loss_aggregated},{metrics_aggregated["accuracy"]}\n')
        
        return aggregated_result

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(LOG_DIR + f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

def evaluate_metrics_aggregation_fn(eval_metrics):
    # Weigh accuracy of each client by number of examples used
    accuracies = [metrics["accuracy"] * num_examples for num_examples, metrics in eval_metrics]
    examples = [num_examples for num_examples, _ in eval_metrics]

    # Aggregate and print custom metric
    accuracy_aggregated = sum(accuracies) / sum(examples)
    return {'accuracy': accuracy_aggregated}

def on_fit_config_fn(rnd):
    return {
        "learning_rate": 0.001,
        "round": rnd,
        "epochs": 3,
    }

def on_evaluate_config_fn(rnd):
    return {
        "round": rnd,
    }

def start_server(n_rounds, n_clients, fraction_fit):
    strategy = FederatedStrategy(min_available_clients=n_clients,
                      min_fit_clients=3,
                      min_eval_clients=3,
                      fraction_fit=fraction_fit,
                      on_fit_config_fn=on_fit_config_fn,
                      on_evaluate_config_fn=on_evaluate_config_fn,
                      evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
    
    history = fl.server.start_server(
                server_address='[::]:5700',
                strategy=strategy,
                config={"num_rounds": n_rounds},
            )

    return history

if __name__ == '__main__':
    history = start_server(n_rounds=100, n_clients=3, fraction_fit=0.5)
    print(history)