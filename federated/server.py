import flwr as fl
from flwr.server.strategy import FedAvg


def start_server(n_rounds, n_clients, fraction_fit):
    strategy = FedAvg(min_available_clients=n_clients,
                      min_fit_clients=3,
                      min_eval_clients=3,
                      fraction_fit=fraction_fit)
    
    history = fl.server.start_server(
                server_address='[::]:5700',
                strategy=strategy,
                config={"num_rounds": n_rounds},
            )

    return history

if __name__ == '__main__':
    history = start_server(n_rounds=3, n_clients=3, fraction_fit=0.5)
    print(history)