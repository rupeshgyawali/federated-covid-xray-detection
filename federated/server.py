import flwr as fl
from flwr.server.strategy import FedAvg


def start_server(n_rounds, n_clients, fraction_fit):
    strategy = FedAvg(min_available_clients=n_clients,
                      fraction_fit=fraction_fit)
    
    fl.server.start_server(
        server_address='[::]:5700',
        strategy=strategy,
        config={"num_rounds": n_rounds},
    )


if __name__ == '__main__':
    start_server(n_rounds=1, n_clients=3, fraction_fit=0.5)