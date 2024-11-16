// Required header files
#include <mpi.h>          // MPI library for distributed computing
#include <vector>         // Standard vector container
#include <random>         // Random number generation
#include <iostream>       // Input/output operations
#include <chrono>        // Time measurement
#include <algorithm>      // STL algorithms
#include <numeric>        // Numeric operations
#include <cmath>         // Mathematical functions
#include <functional>     // Function objects

//==============================================================================
// Event System - For handling and broadcasting events across processes
//==============================================================================

// Enum defining different types of preprocessing events
enum class PreprocessingEvent {
    NOISE_DETECTED,       // When noise is found in data
    CLEANING_COMPLETED,   // When cleaning process is finished
    PROGRESS_UPDATE       // For progress reporting
};

// Structure to hold event-related data
struct EventData {
    int recordIndex;      // Index of the record where event occurred
    int parameterIndex;   // Parameter index in the record
    int recordsProcessed; // Number of records processed so far
};

// Event handler class for managing event callbacks
class EventHandler {
private:
    // Function pointer to store event callback
    std::function<void(PreprocessingEvent, const EventData&)> eventCallback;

public:
    // Set the callback function for event handling
    void setEventCallback(std::function<void(PreprocessingEvent, const EventData&)> callback) {
        eventCallback = callback;
    }

    // Trigger an event with associated data
    void triggerEvent(PreprocessingEvent event, const EventData& data) {
        if (eventCallback) {
            eventCallback(event, data);
        }
    }
};

//==============================================================================
// Data Structures - For tracking preprocessing progress
//==============================================================================

// Structure to track preprocessing progress
struct PreprocessingProgress {
    int processedRecords;    // Counter for processed records
    int totalRecords;        // Total number of records to process
    int cleanedRecords;      // Counter for cleaned records
};

//==============================================================================
// Main Data Preprocessor Class - Core processing logic
//==============================================================================

class DataPreprocessor {
public:
    // Constants defining data dimensions
    static const int NUM_PARAMETERS = 50;    // Number of parameters per record
    static const int CHUNK_SIZE = 1000;      // Size of data chunks for processing
    static const int TOTAL_RECORDS = 1000000 ;  // Total number of records

private:
    std::vector<double> data;        // Storage for data chunk
    int rank;                        // MPI process rank
    int size;                        // Total number of MPI processes
    PreprocessingProgress progress;  // Progress tracking
    EventHandler eventHandler;       // Event handling

    // Calculate median of a vector of values
    double calculateMedian(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        
        if (sorted.size() % 2 == 0) {
            return (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
        }
        return sorted[sorted.size()/2];
    }

    // Clean data by replacing noise with medians
    void cleanData() {
        std::vector<double> medians(NUM_PARAMETERS);
        std::vector<double> means(NUM_PARAMETERS);
        
        // First pass: Calculate medians and means for each parameter
        for (int j = 0; j < NUM_PARAMETERS; j++) {
            std::vector<double> validValues;
            double sum = 0;
            int count = 0;
            
            // Collect valid values for parameter j
            for (int i = 0; i < CHUNK_SIZE; i++) {
                double value = data[i * NUM_PARAMETERS + j];
                if (value != -1) {  // -1 represents noise
                    validValues.push_back(value);
                    sum += value;
                    count++;
                }
            }
            
            // Calculate statistics
            medians[j] = calculateMedian(validValues);
            means[j] = count > 0 ? sum / count : 0.0;
        }

        // Second pass: Replace noisy data with medians
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < NUM_PARAMETERS; j++) {
                int idx = i * NUM_PARAMETERS + j;
                if (data[idx] == -1) {  // If noise detected
                    EventData eventData = {i, j, 0};
                    eventHandler.triggerEvent(PreprocessingEvent::NOISE_DETECTED, eventData);
                    
                    data[idx] = medians[j];  // Replace with median
                    progress.cleanedRecords++;
                }
            }
        }

        // Signal completion of cleaning
        eventHandler.triggerEvent(PreprocessingEvent::CLEANING_COMPLETED, {0, 0, 0});
    }

public:
    // Constructor initializing process-specific data
    DataPreprocessor(int _rank, int _size) : rank(_rank), size(_size) {
        progress = {0, TOTAL_RECORDS, 0};
        data.resize(CHUNK_SIZE * NUM_PARAMETERS);
    }

    // Getter for event handler
    EventHandler& getEventHandler() {
        return eventHandler;
    }

    // Process a chunk of data
    void processChunk() {
        cleanData();
        progress.processedRecords += CHUNK_SIZE;
        
        // Gather total progress across all processes
        int total_processed;
        MPI_Allreduce(&progress.processedRecords, &total_processed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Report progress
        EventData progressData = {0, 0, total_processed};
        eventHandler.triggerEvent(PreprocessingEvent::PROGRESS_UPDATE, progressData);
    }

    // Data access methods
    std::vector<double>& getData() { return data; }
    void setData(const std::vector<double>& newData) { data = newData; }
};

//==============================================================================
// Data Generation Functions - For creating test data
//==============================================================================

// Generate synthetic dataset with noise
std::vector<double> generateFullDataset(int totalRecords, int numParameters) {
    std::vector<double> fullData(totalRecords * numParameters);
    
    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);
    std::uniform_real_distribution<> noise_dis(0, 1);
    
    // Generate data with occasional noise (-1)
    for (int i = 0; i < totalRecords; i++) {
        for (int j = 0; j < numParameters; j++) {
            int idx = i * numParameters + j;
            if (noise_dis(gen) < 0.05) {  // 0.05% chance of noise
                fullData[idx] = -1;
            } else {
                fullData[idx] = dis(gen);
            }
        }
    }
    return fullData;
}

//==============================================================================
// Main Function - Program entry point and MPI orchestration
//==============================================================================

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get process rank and total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create preprocessor instance for this process
    DataPreprocessor preprocessor(rank, size);

    // Set up event handling for this process
    preprocessor.getEventHandler().setEventCallback(
        [rank](PreprocessingEvent event, const EventData& data) {
            switch(event) {
                case PreprocessingEvent::NOISE_DETECTED:
                    std::cout << "Process " << rank << ": Noise detected in record " 
                             << data.recordIndex << ", parameter " << data.parameterIndex 
                             << std::endl;
                    break;
                    
                case PreprocessingEvent::PROGRESS_UPDATE:
                    if (rank == 0) {  // Only root process reports progress
                        std::cout << "Total records processed across all processes: " 
                                 << data.recordsProcessed << std::endl;
                    }
                    break;
                    
                case PreprocessingEvent::CLEANING_COMPLETED:
                    std::cout << "Process " << rank << ": Cleaning completed" 
                             << std::endl;
                    break;
            }
        }
    );
    
    // Calculate work distribution
    const int total_chunks = DataPreprocessor::TOTAL_RECORDS / DataPreprocessor::CHUNK_SIZE;
    const int chunks_per_process = total_chunks / size;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<double> fullData;  // Only used by root process
    std::vector<double> chunk(DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS);
    
    // Process multiple chunks
    for (int chunk_idx = 0; chunk_idx < chunks_per_process; chunk_idx++) {
        if (rank == 0) {  // Root process responsibilities
            if (chunk_idx == 0) {  // Generate data only once
                std::cout << "Generating full dataset...\n";
                fullData = generateFullDataset(DataPreprocessor::TOTAL_RECORDS, 
                                            DataPreprocessor::NUM_PARAMETERS);
            }
            
            // Distribute chunks to all processes
            for (int p = 0; p < size; p++) {
                int global_chunk_idx = chunk_idx * size + p;
                int start_idx = global_chunk_idx * DataPreprocessor::CHUNK_SIZE * 
                              DataPreprocessor::NUM_PARAMETERS;
                
                // Prepare chunk for sending
                std::vector<double> send_chunk(fullData.begin() + start_idx,
                                             fullData.begin() + start_idx + 
                                             DataPreprocessor::CHUNK_SIZE * 
                                             DataPreprocessor::NUM_PARAMETERS);
                
                if (p == 0) {  // Root process keeps its chunk
                    chunk = send_chunk;
                } else {  // Send chunks to other processes
                    MPI_Send(send_chunk.data(),
                           DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                           MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {  // Non-root processes receive their chunks
            MPI_Recv(chunk.data(),
                    DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Process received chunk
        preprocessor.setData(chunk);
        preprocessor.processChunk();
    }

    // Gather final statistics
    int local_processed = preprocessor.getData().size() / DataPreprocessor::NUM_PARAMETERS;
    int total_processed;
    MPI_Reduce(&local_processed, &total_processed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Measure execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();

    // Calculate memory usage
    std::size_t localMemoryUsage = sizeof(DataPreprocessor) + 
                                  chunk.capacity() * sizeof(double);
    if (rank == 0) {
        localMemoryUsage += fullData.capacity() * sizeof(double);
    }

    // Root process gathers and reports final metrics
    if (rank == 0) {
        std::vector<double> allTimes(size);
        std::vector<std::size_t> allMemory(size);

        // Gather timing and memory statistics
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, 
                  allTimes.data(), 1, MPI_DOUBLE, 
                  0, MPI_COMM_WORLD);

        MPI_Gather(&localMemoryUsage, 1, MPI_DOUBLE, 
                  allMemory.data(), 1, MPI_DOUBLE, 
                  0, MPI_COMM_WORLD);
        
        // Calculate final statistics
        double totalTime = *std::max_element(allTimes.begin(), allTimes.end());
        double avgTime = std::accumulate(allTimes.begin(), allTimes.end(), 0.0) / size;
        std::size_t totalMemory = std::accumulate(allMemory.begin(), allMemory.end(), 0ULL);

        // Print final performance metrics
        std::cout << "\nFinal Performance Metrics:\n"
                  << "Total Processing Time: " << totalTime << " seconds\n"
                  << "Average Time per Process: " << avgTime << " seconds\n"
                  << "Total Memory Usage: " << totalMemory << " bytes\n"
                  << "Total chunks processed: " << total_chunks << "\n"
                  << "Chunks per process: " << chunks_per_process << "\n";
    } else {
        // Non-root processes just send their data
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&localMemoryUsage, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Cleanup MPI environment
    MPI_Finalize();
    return 0;
}
