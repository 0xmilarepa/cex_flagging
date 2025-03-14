import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community


def calculate_volume_metrics(
    df,
    addresses,
    from_address_col='FROM_ADDRESS',
    to_address_col='TO_ADDRESS',
    origin_from_col=None,  # Optional: Column for origin_from_address
    origin_to_col=None,    # Optional: Column for origin_to_address
    amount_col='AMOUNT_USD',
    contract_address_col=None  # Optional: Column for contract_address
):
    """
    This function calculates the average outgoing, incoming, and total volume metrics for wallet addresses in a determined DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data.
        addresses (list): List of addresses to analyze.
        from_address_col (str): Name of the column containing sender addresses. Default is 'FROM_ADDRESS'.
        to_address_col (str): Name of the column containing receiver addresses. Default is 'TO_ADDRESS'.
        origin_from_col (str): Optional. Name of the column containing origin_from_address. Default is None.
        origin_to_col (str): Optional. Name of the column containing origin_to_address. Default is None.
        amount_col (str): Name of the column containing transaction amounts. Default is 'AMOUNT_USD'.
        contract_address_col (str): Optional. Name of the column containing contract_address. Default is None.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - address: The address.
            - outgoing_volume_USD: Average outgoing volume for each address.
            - incoming_volume_USD: Average incoming volume for each address.
            - total_volume_USD: Sum of average incoming and outgoing volumes.
    """
    # Combine all relevant addresses into a single Series and drop duplicates
    all_address_columns = [from_address_col, to_address_col]
    if origin_from_col and origin_from_col in df.columns:
        all_address_columns.append(origin_from_col)
    if origin_to_col and origin_to_col in df.columns:
        all_address_columns.append(origin_to_col)
    if contract_address_col and contract_address_col in df.columns:
        all_address_columns.append(contract_address_col)

    all_addresses = pd.concat([df[col] for col in all_address_columns]).drop_duplicates().dropna().unique()

    # Filter addresses to only those in the provided list
    filtered_addresses = [addr for addr in all_addresses if addr in addresses]

    # Initialize a dictionary to store results
    results = {addr: {'outgoing_volume_USD': 0, 'incoming_volume_USD': 0, 'total_volume_USD': 0} for addr in filtered_addresses}

    # Calculate average outgoing volume for each address when it is the sender
    sender_averages = df[
        df[from_address_col].isin(filtered_addresses)
    ].groupby(from_address_col)[amount_col].mean()

    # Update results with sender averages
    for addr, avg in sender_averages.items():
        results[addr]['outgoing_volume_USD'] = avg

    # Calculate average incoming volume for each address when it is the receiver
    receiver_averages = df[
        df[to_address_col].isin(filtered_addresses)
    ].groupby(to_address_col)[amount_col].mean()

    # Update results with receiver averages
    for addr, avg in receiver_averages.items():
        results[addr]['incoming_volume_USD'] = avg

    # Calculate average volume for origin_from addresses (if column exists)
    if origin_from_col and origin_from_col in df.columns:
        origin_from_averages = df[
            df[origin_from_col].isin(filtered_addresses)
        ].groupby(origin_from_col)[amount_col].mean()

        # Update results with origin_from averages
        for addr, avg in origin_from_averages.items():
            results[addr]['outgoing_volume_USD'] += avg

    # Calculate average volume for origin_to addresses (if column exists)
    if origin_to_col and origin_to_col in df.columns:
        origin_to_averages = df[
            df[origin_to_col].isin(filtered_addresses)
        ].groupby(origin_to_col)[amount_col].mean()

        # Update results with origin_to averages
        for addr, avg in origin_to_averages.items():
            results[addr]['incoming_volume_USD'] += avg

    # Calculate average volume for contract addresses (if column exists)
    if contract_address_col and contract_address_col in df.columns:
        contract_averages = df[
            df[contract_address_col].isin(filtered_addresses)
        ].groupby(contract_address_col)[amount_col].mean()

        # Update results with contract averages
        for addr, avg in contract_averages.items():
            results[addr]['outgoing_volume_USD'] += avg

    # Calculate total_volume_USD for each address
    for addr in results:
        results[addr]['total_volume_USD'] = (
            results[addr]['outgoing_volume_USD'] + results[addr]['incoming_volume_USD']
        )

    # Convert the results dictionary to a DataFrame
    volume_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'address'})

    return volume_df

################################################################################

def calculate_unique_interactions(
    df,
    addresses,
    from_address_col='FROM_ADDRESS',
    to_address_col='TO_ADDRESS',
    origin_from_col=None,  # Optional: Column for origin_from_address
    origin_to_col=None,    # Optional: Column for origin_to_address
    contract_address_col=None  # Optional: Column for contract_address
):
    """
    This function calculates the number of unique interactions per address.

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data.
        addresses (list): List of addresses to analyze.
        from_address_col (str): Name of the column containing sender addresses. Default is 'FROM_ADDRESS'.
        to_address_col (str): Name of the column containing receiver addresses. Default is 'TO_ADDRESS'.
        origin_from_col (str): Optional. Name of the column containing origin_from_address. Default is None.
        origin_to_col (str): Optional. Name of the column containing origin_to_address. Default is None.
        contract_address_col (str): Optional. Name of the column containing contract_address. Default is None.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - address: The address.
            - unique_interactions: Number of unique interactions per address.
    """
    # Initialize a dictionary to store interaction counts
    interactions = {}

    for addr in addresses:
        # Initialize a set to store all unique interactions
        all_interactions = set()

        # When address is receiver (TO_ADDRESS)
        receiver_interactions = df[df[to_address_col] == addr][from_address_col].unique()
        receiver_interactions = [addr for addr in receiver_interactions if addr not in addresses]
        all_interactions.update(receiver_interactions)

        # When address is sender (FROM_ADDRESS)
        sender_interactions = df[df[from_address_col] == addr][to_address_col].unique()
        sender_interactions = [addr for addr in sender_interactions if addr not in addresses]
        all_interactions.update(sender_interactions)

        # When address is origin_from (ORIGIN_FROM_ADDRESS)
        if origin_from_col and origin_from_col in df.columns:
            origin_from_interactions = df[df[origin_from_col] == addr][from_address_col].unique()
            origin_from_interactions = [addr for addr in origin_from_interactions if addr not in addresses]
            all_interactions.update(origin_from_interactions)

        # When address is origin_to (ORIGIN_TO_ADDRESS)
        if origin_to_col and origin_to_col in df.columns:
            origin_to_interactions = df[df[origin_to_col] == addr][to_address_col].unique()
            origin_to_interactions = [addr for addr in origin_to_interactions if addr not in addresses]
            all_interactions.update(origin_to_interactions)

        # When address is contract_address (CONTRACT_ADDRESS)
        if contract_address_col and contract_address_col in df.columns:
            contract_interactions = df[df[contract_address_col] == addr][from_address_col].unique()
            contract_interactions = [addr for addr in contract_interactions if addr not in addresses]
            all_interactions.update(contract_interactions)

        # Count unique interactions
        interactions[addr] = len(all_interactions)

    # Convert the dictionary to a DataFrame
    interaction_df = pd.DataFrame({
        'address': list(interactions.keys()),
        'unique_interactions': list(interactions.values())
    })

    return interaction_df

################################################################################

def calculate_activity(
    df,
    addresses,
    from_address_col='FROM_ADDRESS',
    to_address_col='TO_ADDRESS',
    origin_from_col=None,  # Optional: Column for origin_from_address
    origin_to_col=None,  # Optional: Column for origin_to_address
    timestamp_col=None,  # Optional: Column for transaction timestamps
    contract_address_col=None  # Optional: Column for contract_address
):
    """
    This function calculates the activity metrics for a list of addresses.

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data.
        addresses (list): List of addresses to analyze.
        from_address_col (str): Name of the column containing sender addresses. Default is 'FROM_ADDRESS'.
        to_address_col (str): Name of the column containing receiver addresses. Default is 'TO_ADDRESS'.
        origin_from_col (str): Optional. Name of the column containing origin_from_address. Default is None.
        origin_to_col (str): Optional. Name of the column containing origin_to_address. Default is None.
        timestamp_col (str): Optional. Name of the column containing transaction timestamps. Default is None.
        contract_address_col (str): Optional. Name of the column containing contract_address. Default is None.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - address: The address.
            - tx_count: Number of transactions involving the address.
            - tx_per_hour: Average number of transactions per hour (if timestamp_col is provided).
            - active_days: Duration of activity in days (last_tx - first_tx).
    """
    # Combine all relevant addresses into a single Series
    all_address_columns = [from_address_col, to_address_col]
    if origin_from_col and origin_from_col in df.columns:
        all_address_columns.append(origin_from_col)
    if origin_to_col and origin_to_col in df.columns:
        all_address_columns.append(origin_to_col)
    if contract_address_col and contract_address_col in df.columns:
        all_address_columns.append(contract_address_col)

    all_addresses = pd.concat([df[col] for col in all_address_columns]).dropna().unique()

    # Filter addresses to only those in the provided list
    filtered_addresses = [addr for addr in all_addresses if addr in addresses]

    # Initialize a list to store results
    activity_data = []

    for addr in filtered_addresses:
        # Filter transactions involving the address
        addr_txs = df[
            (df[from_address_col] == addr) |
            (df[to_address_col] == addr) |
            (df[origin_from_col] == addr if origin_from_col and origin_from_col in df.columns else False) |
            (df[origin_to_col] == addr if origin_to_col and origin_to_col in df.columns else False) |
            (df[contract_address_col] == addr if contract_address_col and contract_address_col in df.columns else False)
        ]

        # Calculate metrics
        tx_count = addr_txs.shape[0]

        # Calculate first_tx, last_tx, and active_duration if timestamp_col is provided
        if timestamp_col and timestamp_col in df.columns:
            first_tx = addr_txs[timestamp_col].min()
            last_tx = addr_txs[timestamp_col].max()
            active_days = (last_tx - first_tx).total_seconds() / (60 * 60 * 24)  # Convert to days

            # If active_days is 0, set it to 1
            if active_days == 0:
                active_days = 1

            # Calculate tx_per_hour
            tx_per_hour = addr_txs.groupby(addr_txs[timestamp_col].dt.floor('h')).size().mean()
        else:
            first_tx = None
            last_tx = None
            active_days = None
            tx_per_hour = None

        # Append to the list
        activity_data.append({
            'address': addr,
            'tx_count': tx_count,
            'tx_per_hour': tx_per_hour,
            'active_days': active_days,
        })

    # Convert the list to a DataFrame
    activity_df = pd.DataFrame(activity_data)

    return activity_df

################################################################################

def identify_clusters(
    df,
    addresses,
    from_address_col='FROM_ADDRESS',
    to_address_col='TO_ADDRESS',
    min_cluster_size=100,  # Minimum size of a cluster to be considered "large"
    resolution=1.0  # Resolution parameter for Louvain
):
    """
    This function identifies addresses that are part of large clusters using the Louvain community detection method.

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data.
        from_address_col (str): Name of the column containing sender addresses. Default is 'FROM_ADDRESS'.
        to_address_col (str): Name of the column containing receiver addresses. Default is 'TO_ADDRESS'.
        min_cluster_size (int): Minimum size of a cluster to be considered "large". Default is 100.
        resolution (float): Resolution parameter for Louvain. Higher values result in smaller communities. Default is 1.0.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - address: The address.
            - is_large_cluster: 1 if the address is part of a large cluster, 0 otherwise.
    """

    # Create a directed graph from the transaction data
    G = nx.from_pandas_edgelist(
        df,
        source=from_address_col,
        target=to_address_col,
        create_using=nx.DiGraph()
    )

    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()

    # Find communities using the Louvain method
    communities = community.louvain_communities(G_undirected, resolution=resolution)

    # Initialize a dictionary to store cluster labels
    cluster_labels = {addr: 0 for addr in addresses}

    # Label addresses in large communities
    for comm in communities:
        if len(comm) >= min_cluster_size:
            for addr in addresses:
                if addr in comm:
                    cluster_labels[addr] = 1

    # Convert the dictionary to a DataFrame
    cluster_df = pd.DataFrame({
        'address': list(cluster_labels.keys()),
        'is_large_cluster': list(cluster_labels.values())
    })

    return cluster_df

################################################################################

def cluster_sizes(
    df,
    addresses,  # List of addresses to check
    from_address_col='FROM_ADDRESS',
    to_address_col='TO_ADDRESS',
    resolution=1.0  # Resolution parameter for Louvain
):
    """
    This function calculates the size of the clusters that specific addresses belong to.

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data.
        addresses (list): List of addresses to check for cluster membership.
        from_address_col (str): Name of the column containing sender addresses. Default is 'FROM_ADDRESS'.
        to_address_col (str): Name of the column containing receiver addresses. Default is 'TO_ADDRESS'.
        resolution (float): Resolution parameter for Louvain. Higher values result in smaller communities. Default is 1.0.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - address: The address.
            - cluster_size: Size of the cluster the address belongs to (0 if not part of any cluster).
    """
    # Create a directed graph from the transaction data
    G = nx.from_pandas_edgelist(
        df,
        source=from_address_col,
        target=to_address_col,
        create_using=nx.DiGraph()
    )

    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()

    # Find communities using the Louvain method
    communities = community.louvain_communities(G_undirected, resolution=resolution)

    # Initialize a dictionary to store cluster sizes for the addresses to check
    cluster_sizes = {addr: 0 for addr in addresses}

    # Check the cluster size for each address
    for comm in communities:
        for addr in addresses:
            if addr in comm:
                cluster_sizes[addr] = len(comm)

    # Convert the dictionary to a DataFrame
    cluster_size_df = pd.DataFrame({
        'address': list(cluster_sizes.keys()),
        'cluster_size': list(cluster_sizes.values())
    })

    return cluster_size_df

################################################################################

def pipeline_processing(
    df,
    addresses,
    from_address_col='FROM_ADDRESS',
    to_address_col='TO_ADDRESS',
    origin_from_col=None,  # Optional: Column for origin_from_address
    origin_to_col=None,    # Optional: Column for origin_to_address
    amount_col='AMOUNT_USD',  # Column for transaction amounts
    timestamp_col=None,  # Optional: Column for transaction timestamps
    contract_address_col=None,  # Optional: Column for contract_address
    min_cluster_size=500,  # Minimum size of a cluster to be considered "large"
    resolution=1.0  # Resolution parameter for Louvain
):
    """
    This is a pipeline function. It takes a DataFrame based on list of addresses and processes it through all metrics.

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data.
        addresses (list): List of addresses to analyze.
        from_address_col (str): Name of the column containing sender addresses. Default is 'FROM_ADDRESS'.
        to_address_col (str): Name of the column containing receiver addresses. Default is 'TO_ADDRESS'.
        origin_from_col (str): Optional. Name of the column containing origin_from_address. Default is None.
        origin_to_col (str): Optional. Name of the column containing origin_to_address. Default is None.
        amount_col (str): Name of the column containing transaction amounts. Default is 'AMOUNT_USD'.
        timestamp_col (str): Optional. Name of the column containing transaction timestamps. Default is None.
        contract_address_col (str): Optional. Name of the column containing contract_address. Default is None.
        min_cluster_size (int): Minimum size of a cluster to be considered "large". Default is 500.
        resolution (float): Resolution parameter for Louvain. Higher values result in smaller communities. Default is 1.0.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - address: The address.
            - prediction label: 1 for CEX/Bridge, 0 for other.
            - outgoing_volume_USD: Average outgoing volume for each address.
            - incoming_volume_USD: Average incoming volume for each address.
            - total_volume_USD: Sum of average incoming and outgoing volumes.
            - unique_interactions: Number of unique interactions per address.
            - tx_count: Number of transactions involving the address.
            - first_tx: Timestamp of the first transaction (if timestamp_col is provided).
            - last_tx: Timestamp of the last transaction (if timestamp_col is provided).
            - tx_per_hour: Average number of transactions per hour (if timestamp_col is provided).
            - is_large_cluster: 1 if the address is part of a large cluster, 0 otherwise.
            - interaction_volume: Interaction volume (unique_interactions * total_volume_USD).
    """
    # Step 1: Calculate volume metrics
    volume_df = calculate_volume_metrics(
        df,
        addresses,
        from_address_col=from_address_col,
        to_address_col=to_address_col,
        origin_from_col=origin_from_col,
        origin_to_col=origin_to_col,
        amount_col=amount_col,
        contract_address_col=contract_address_col
    )
    print("Volume processing done ✅")

    # Step 2: Calculate unique interactions
    interaction_df = calculate_unique_interactions(
        df,
        addresses,
        from_address_col=from_address_col,
        to_address_col=to_address_col,
        origin_from_col=origin_from_col,
        origin_to_col=origin_to_col,
        contract_address_col=contract_address_col
    )
    print("Unique interactions processing done ✅")

    # Step 3: Calculate activity metrics
    activity_df = calculate_activity(
        df,
        addresses,
        from_address_col=from_address_col,
        to_address_col=to_address_col,
        origin_from_col=origin_from_col,
        origin_to_col=origin_to_col,
        timestamp_col=timestamp_col,
        contract_address_col=contract_address_col
    )

    print("Activity processing done ✅")

    # Step 4: Identify clusters
    cluster_df = identify_clusters(
        df,
        addresses,
        from_address_col=from_address_col,
        to_address_col=to_address_col,
        min_cluster_size=min_cluster_size,
        resolution=resolution
    )

    print("Cluster processing done ✅")

    # Step 5: Merge all results into a single DataFrame
    result_df = volume_df.merge(interaction_df, on='address', how='left')
    result_df = result_df.merge(activity_df, on='address', how='left')
    result_df = result_df.merge(cluster_df, on='address', how='left')

    # Step 6: Calculate interaction_volume
    result_df['interaction_volume_USD'] = result_df['unique_interactions'] * result_df['total_volume_USD']

    print("All done ✅")

    return result_df
