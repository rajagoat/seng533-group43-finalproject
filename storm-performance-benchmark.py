#!/usr/bin/env python3
"""
Apache Storm Performance Benchmark - Fixed Version
---------------------------------
This script benchmarks Apache Storm performance using a flight delay prediction topology.
It tests different configurations and collects metrics comparable to the Spark benchmark.
"""

import os
import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import argparse
import json
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameter values for iteration (similar to Spark benchmarks)
WORKER_COUNT_VALUES = [1, 2, 4]      # Equivalent to num_cores in Spark
WORKER_MEMORY_VALUES = ["2g", "4g", "8g"]  # Worker memory sizes
PARALLELISM_VALUES = [2, 4, 8]       # Parallelism levels

# Dataset paths - adjust these to your local paths
DATASETS = {
    "1k": "Datasets/flights_sample_1k.csv",
    "500k": "Datasets/flights_sample_500k.csv",
    "3m": "Datasets/flights_sample_3m.csv"
}

# Results file
EXCEL_FILENAME = "storm_experiment_results.xlsx"

# Columns for results dataframe
COLUMNS = [
    "worker_count",
    "worker_memory",
    "parallelism",
    "tuple_processing_rate",
    "file_size",
    "train_duration_s",
    "train_count",
    "train_throughput_records_s",
    "pred_duration_s",
    "test_count",
    "pred_throughput_records_s",
    "rmse",
    "r2",
    "cpu_usage_start_percent",
    "cpu_usage_end_percent",
    "mem_used_at_start_mb",
    "mem_used_at_end_mb",
    "timestamp"
]

class FlightDataSpout:
    """
    Spout that reads flight data from CSV file and emits tuples.
    """
    def __init__(self, data_file, tuple_rate=1000, batch_size=100):
        self.data_file = data_file
        self.tuple_rate = tuple_rate
        self.batch_size = batch_size
        self.output_queue = queue.Queue()
        self.metrics = {}
        
    def load_data(self):
        """Load and preprocess data from CSV file"""
        logger.info(f"Loading data from {self.data_file}")
        
        # Read CSV file
        df = pd.read_csv(self.data_file)
        
        # Filter out canceled or diverted flights and drop rows with null arrival delay
        df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
        df = df.dropna(subset=["ARR_DELAY"])
        
        # Fill missing weather delay values with 0
        df = df.fillna({"DELAY_DUE_WEATHER": 0})
        
        # Add weather condition feature
        df["WeatherCond"] = df["DELAY_DUE_WEATHER"].apply(lambda x: "Bad" if x > 0 else "Normal")
        
        # Split into train and test
        train_size = int(len(df) * 0.8)
        self.train_df = df[:train_size]
        self.test_df = df[train_size:]
        
        # Record counts
        self.metrics["train_count"] = len(self.train_df)
        self.metrics["test_count"] = len(self.test_df)
        
        logger.info(f"Data loaded: {len(self.train_df)} training records, {len(self.test_df)} test records")
    
    def run(self):
        """Run the spout, emitting data at the specified rate"""
        # Load data first
        self.load_data()
        
        # Emit training data
        logger.info("Emitting training data...")
        train_start_time = time.time()
        
        for i in range(0, len(self.train_df), self.batch_size):
            batch = self.train_df.iloc[i:i+self.batch_size]
            batch_dict = batch.to_dict('records')
            self.output_queue.put(("train_data", batch_dict))
            
            # Control emission rate
            if self.tuple_rate > 0:
                sleep_time = self.batch_size / self.tuple_rate
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Signal end of training data
        self.output_queue.put(("control", "END_OF_TRAINING"))
        
        # Record training duration
        train_duration = time.time() - train_start_time
        self.metrics["train_duration_s"] = train_duration
        self.metrics["train_throughput_records_s"] = len(self.train_df) / train_duration
        
        # Emit testing data
        logger.info("Emitting testing data...")
        test_start_time = time.time()
        
        for i in range(0, len(self.test_df), self.batch_size):
            batch = self.test_df.iloc[i:i+self.batch_size]
            batch_dict = batch.to_dict('records')
            self.output_queue.put(("test_data", batch_dict))
            
            # Control emission rate
            if self.tuple_rate > 0:
                sleep_time = self.batch_size / self.tuple_rate
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Signal end of test data
        self.output_queue.put(("control", "END_OF_TESTING"))
        
        # Record prediction duration
        pred_duration = time.time() - test_start_time
        self.metrics["pred_duration_s"] = pred_duration
        self.metrics["pred_throughput_records_s"] = len(self.test_df) / pred_duration
        
        # Signal completion
        self.output_queue.put(("metrics", self.metrics))
        self.output_queue.put(("control", "DONE"))
        
        logger.info("Spout finished emitting all data")

class FeatureProcessingBolt:
    """
    Bolt that processes features for the flight data.
    """
    def __init__(self, input_queue, output_queue, parallelism=1):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.parallelism = parallelism
        self.processed_count = 0
        self.running = True
        self.workers = []
        self.stop_event = threading.Event()
    
    def process_batch(self, batch):
        """Process a batch of records"""
        processed_batch = []
        
        for record in batch:
            # Extract features (same as in Spark job)
            features = {
                "DISTANCE": record["DISTANCE"],
                "CRS_DEP_TIME": record["CRS_DEP_TIME"],
                "WeatherCond": record["WeatherCond"],
                "ARR_DELAY": record["ARR_DELAY"]
            }
            processed_batch.append(features)
        
        return processed_batch
    
    def worker_loop(self, worker_id):
        """Worker loop for parallel processing"""
        logger.info(f"Feature processing worker {worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get tuple from input queue with timeout
                message = self.input_queue.get(timeout=0.5)
                
                if message is None:
                    continue
                    
                msg_type, content = message
                
                if msg_type == "control":
                    # Pass through control messages
                    self.output_queue.put(message)
                    if content == "DONE":
                        break
                elif msg_type in ["train_data", "test_data"]:
                    # Process batch
                    batch = content
                    processed_batch = self.process_batch(batch)
                    self.processed_count += len(processed_batch)
                    
                    # Emit processed batch
                    self.output_queue.put((msg_type, processed_batch))
                    
                    if self.processed_count % 1000 == 0:
                        logger.info(f"Processed {self.processed_count} records")
                        
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in feature processing worker {worker_id}: {str(e)}")
        
        logger.info(f"Feature processing worker {worker_id} stopped")
    
    def run(self):
        """Run the bolt with multiple workers"""
        # Start worker threads
        for i in range(self.parallelism):
            worker = threading.Thread(target=self.worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop all worker threads"""
        logger.info("Stopping feature processing workers")
        self.stop_event.set()
        
        # Wait for threads to finish gracefully (with timeout)
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        logger.info(f"Feature processing bolt stopped. Processed {self.processed_count} records.")

class ModelTrainingBolt:
    """
    Bolt that trains a linear regression model.
    """
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.training_data = []
        self.testing_data = []
        self.model = None
        self.metrics = {}
        self.running = True
        self.stop_event = threading.Event()
    
    def train_model(self):
        """Train a linear regression model on the collected data"""
        if not self.training_data:
            logger.warning("No training data available")
            return False
        
        logger.info(f"Training model on {len(self.training_data)} records")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.training_data)
            
            # Prepare features
            categorical_features = ['WeatherCond']
            numeric_features = ['DISTANCE', 'CRS_DEP_TIME']
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features),
                    ('num', 'passthrough', numeric_features)
                ])
            
            # Create and train the pipeline
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
            X = df[['DISTANCE', 'CRS_DEP_TIME', 'WeatherCond']]
            y = df['ARR_DELAY']
            
            self.pipeline.fit(X, y)
            
            logger.info("Model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def make_predictions(self):
        """Make predictions on testing data and calculate metrics"""
        if not self.model or not self.testing_data:
            logger.warning("No model or testing data available")
            return
        
        logger.info(f"Making predictions on {len(self.testing_data)} records")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.testing_data)
            
            X = df[['DISTANCE', 'CRS_DEP_TIME', 'WeatherCond']]
            y_true = df['ARR_DELAY']
            
            # Make predictions
            y_pred = self.pipeline.predict(X)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # Store metrics
            self.metrics.update({
                "rmse": rmse,
                "r2": r2
            })
            
            logger.info(f"Predictions made. RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Clear testing data to free memory
            self.testing_data = []
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
    
    def run(self):
        """Run the model training bolt"""
        training_completed = False
        testing_completed = False
        
        while not self.stop_event.is_set():
            try:
                # Get tuple from input queue with timeout
                message = self.input_queue.get(timeout=0.5)
                
                if message is None:
                    continue
                    
                msg_type, content = message
                
                if msg_type == "control":
                    if content == "END_OF_TRAINING":
                        logger.info("End of training data received")
                        # Train the model
                        if self.train_model():
                            self.model = True  # Indicate model is trained
                            training_completed = True
                    elif content == "END_OF_TESTING":
                        logger.info("End of testing data received")
                        if self.model:
                            self.make_predictions()
                        testing_completed = True
                    elif content == "DONE":
                        # Pass metrics and done message
                        self.output_queue.put(("metrics", self.metrics))
                        self.output_queue.put(message)  # Pass through DONE message
                        break
                    else:
                        # Pass through other control messages
                        self.output_queue.put(message)
                        
                elif msg_type == "train_data" and not training_completed:
                    # Collect training data
                    self.training_data.extend(content)
                    
                elif msg_type == "test_data" and training_completed and not testing_completed:
                    # Collect testing data
                    self.testing_data.extend(content)
                    
                    # Make predictions if we have accumulated enough data
                    if len(self.testing_data) >= 1000:
                        self.make_predictions()
                
                elif msg_type == "metrics":
                    # Merge metrics
                    self.metrics.update(content)
                
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in model training bolt: {str(e)}")
        
        logger.info("Model training bolt stopped")
    
    def stop(self):
        """Stop the model training thread"""
        logger.info("Stopping model training bolt")
        self.stop_event.set()

def run_experiment(data_file, worker_count, worker_memory, parallelism, tuple_rate=1000):
    """
    Run a single experiment with the given configuration.
    
    Args:
        data_file: Path to the flight data CSV file
        worker_count: Number of Storm workers to use
        worker_memory: Memory allocation for each worker
        parallelism: Parallelism hint for spouts and bolts
        tuple_rate: Target tuples per second emission rate
    
    Returns:
        Dictionary with experiment results
    """
    # Determine file size category
    if "1k" in data_file.lower():
        file_size = "1k"
    elif "500k" in data_file.lower():
        file_size = "500k"
    elif "3m" in data_file.lower():
        file_size = "3m"
    else:
        file_size = "unknown"
    
    logger.info(f"Starting experiment with: workers={worker_count}, memory={worker_memory}, " 
               f"parallelism={parallelism}, file_size={file_size}")
    
    # Record system stats before starting
    cpu_usage_start = psutil.cpu_percent(interval=1)
    mem_info_start = psutil.virtual_memory()
    mem_used_start_mb = mem_info_start.used / (1024 * 1024)
    
    # Simulate worker memory allocation (just for metrics, doesn't actually limit memory)
    memory_gb = int(worker_memory.lower().replace('g', ''))
    
    # Set up communication queues
    spout_to_bolt_queue = queue.Queue()
    bolt_to_model_queue = queue.Queue()
    model_to_output_queue = queue.Queue()
    
    # Create topology components
    spout = FlightDataSpout(data_file, tuple_rate=tuple_rate)
    processing_bolt = FeatureProcessingBolt(spout_to_bolt_queue, bolt_to_model_queue, parallelism=parallelism)
    model_bolt = ModelTrainingBolt(bolt_to_model_queue, model_to_output_queue)
    
    # Start bolts first (they wait for data)
    logger.info("Starting topology threads")
    processing_bolt.run()
    
    model_thread = threading.Thread(target=lambda: model_bolt.run())
    model_thread.daemon = True
    model_thread.start()
    
    # Start spout (data producer)
    spout_thread = threading.Thread(target=lambda: spout.run())
    spout_thread.daemon = True
    spout_thread.start()
    
    # Move data from spout to processing bolt
    done_received = False
    try:
        while not done_received:
            try:
                message = spout.output_queue.get(timeout=0.5)
                spout_to_bolt_queue.put(message)
                
                msg_type, content = message
                if msg_type == "control" and content == "DONE":
                    done_received = True
                    
                # Mark task as done
                spout.output_queue.task_done()
            except queue.Empty:
                if not spout_thread.is_alive():
                    # Add a DONE message if spout thread died without sending one
                    if not done_received:
                        spout_to_bolt_queue.put(("control", "DONE"))
                        done_received = True
    except Exception as e:
        logger.error(f"Error in main thread: {str(e)}")
    
    # Wait for spout thread to finish
    logger.info("Waiting for threads to finish")
    spout_thread.join(timeout=5.0)
    
    # Collect all metrics from output queue (non-blocking)
    metrics = {}
    done_received = False
    
    # Time limit for waiting for results (10 seconds)
    end_time = time.time() + 10
    
    while time.time() < end_time and not done_received:
        try:
            message = model_to_output_queue.get(timeout=0.5)
            
            if message is None:
                continue
                
            msg_type, content = message
            
            if msg_type == "metrics":
                metrics.update(content)
            elif msg_type == "control" and content == "DONE":
                done_received = True
                
            # Mark task as done
            model_to_output_queue.task_done()
        except queue.Empty:
            continue
    
    # Stop all components
    logger.info("Stopping components")
    processing_bolt.stop()
    model_bolt.stop()
    
    # Add spout metrics
    metrics.update(spout.metrics)
    
    # Record system stats after completion
    cpu_usage_end = psutil.cpu_percent(interval=1)
    mem_info_end = psutil.virtual_memory()
    mem_used_end_mb = mem_info_end.used / (1024 * 1024)
    
    # Add configuration and system metrics
    metrics.update({
        "worker_count": worker_count,
        "worker_memory": worker_memory,
        "parallelism": parallelism,
        "tuple_processing_rate": tuple_rate,
        "file_size": file_size,
        "cpu_usage_start_percent": cpu_usage_start,
        "cpu_usage_end_percent": cpu_usage_end,
        "mem_used_at_start_mb": mem_used_start_mb,
        "mem_used_at_end_mb": mem_used_end_mb,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    logger.info(f"Experiment completed. Metrics: {metrics}")
    
    return metrics

def main():
    """
    Main function to run the Storm performance benchmark.
    """
    parser = argparse.ArgumentParser(description='Apache Storm Performance Benchmark')
    parser.add_argument('--dataset', choices=['1k', '500k', '3m'], default='1k',
                        help='Dataset size to use')
    parser.add_argument('--output', default=EXCEL_FILENAME,
                        help='Output Excel file for results')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run a quick test with just a few configurations')
    args = parser.parse_args()
    
    # Get dataset path
    data_file = DATASETS[args.dataset]
    
    # Check if dataset exists
    if not os.path.exists(data_file):
        logger.error(f"Dataset file not found: {data_file}")
        return
    
    # Check if results file exists
    if os.path.exists(args.output):
        results_df = pd.read_excel(args.output)
        logger.info(f"Loaded existing results from {args.output} with {len(results_df)} records")
    else:
        results_df = pd.DataFrame(columns=COLUMNS)
        logger.info(f"Creating new results DataFrame")
    
    # If quick test, use only one value for each parameter
    if args.quick_test:
        worker_counts = [2]
        worker_memories = ["4g"]
        parallelism_values = [4]
    else:
        worker_counts = WORKER_COUNT_VALUES
        worker_memories = WORKER_MEMORY_VALUES
        parallelism_values = PARALLELISM_VALUES
    
    # Run experiments for all configurations
    for worker_count, worker_memory, parallelism in product(
            worker_counts, worker_memories, parallelism_values):
        
        # Skip if this configuration was already tested
        if len(results_df) > 0:
            existing = results_df[(results_df['worker_count'] == worker_count) &
                               (results_df['worker_memory'] == worker_memory) &
                               (results_df['parallelism'] == parallelism) &
                               (results_df['file_size'] == args.dataset)]
            if len(existing) > 0:
                logger.info(f"Skipping already tested configuration: workers={worker_count}, "
                           f"memory={worker_memory}, parallelism={parallelism}")
                continue
        
        # Run experiment
        logger.info(f"Running experiment with: workers={worker_count}, memory={worker_memory}, "
                   f"parallelism={parallelism}")
        
        # Define tuple rates based on the parallelism (similar to batch interval in Spark)
        tuple_rate = 1000 * parallelism  # Scale tuple rate with parallelism
        
        metrics = run_experiment(data_file, worker_count, worker_memory, parallelism, tuple_rate)
        
        if metrics:
            # Add to results DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
            
            # Save results after each experiment
            results_df.to_excel(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
        else:
            logger.error(f"Experiment failed: workers={worker_count}, memory={worker_memory}, "
                       f"parallelism={parallelism}")
    
    # Generate visualizations if we have results
    if len(results_df) > 0:
        try:
            from visualize_results import generate_visualizations
            generate_visualizations(args.output)
        except ImportError:
            logger.info("Visualization module not found. Skipping visualization generation.")
    
    logger.info("Benchmark completed")

if __name__ == "__main__":
    main()