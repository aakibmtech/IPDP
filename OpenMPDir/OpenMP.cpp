// Required header files
#include <omp.h>           // OpenMP parallel programming support
#include <vector>          // STL vector container
#include <random>          // Random number generation
#include <iostream>        // Input/output operations
#include <chrono>         // Time measurement
#include <algorithm>       // STL algorithms
#include <numeric>         // Numeric operations
#include <cmath>          // Mathematical functions
#include <functional>      // Function objects
#include <mutex>          // Mutual exclusion
#include <atomic>         // Atomic operations

//==============================================================================
// Event System - Handles notifications and callbacks for preprocessing events
//==============================================================================

// Defines different types of events that can occur during preprocessing
enum class PreprocessingEvent {
    NOISE_DETECTED,        // When noise is found in the data
    CLEANING_COMPLETED,    // When a cleaning operation is finished
    PROGRESS_UPDATE        // When progress needs to be reported
};

// Structure to hold event-related data
struct EventData {
    int recordIndex;       // Index of the record being processed
    int parameterIndex;    // Index of the parameter being processed
    int recordsProcessed;  // Total number of records processed so far
};

// Event handler class to manage event callbacks
class EventHandler {
private:
    // Function object to store the callback function
    std::function<void(PreprocessingEvent, const EventData&)> eventCallback;
    std::mutex eventMutex;  // Mutex for thread-safe event handling

public:
    // Sets the callback function for event handling
    void setEventCallback(std::function<void(PreprocessingEvent, const EventData&)> callback) {
        std::lock_guard<std::mutex> lock(eventMutex);
        eventCallback = callback;
    }

    // Triggers an event with associated data
    void triggerEvent(PreprocessingEvent event, const EventData& data) {
        std::lock_guard<std::mutex> lock(eventMutex);
        if (eventCallback) {
            eventCallback(event, data);
        }
    }
};

//==============================================================================
// Data Structures - Holds progress information and tracking data
//==============================================================================

// Structure to track preprocessing progress
struct PreprocessingProgress {
    std::atomic<int> processedRecords;  // Atomic counter for processed records
    int totalRecords;                   // Total number of records to process
    std::atomic<int> cleanedRecords;    // Atomic counter for cleaned records
    
    // Constructor initializes all counters to zero
    PreprocessingProgress() : processedRecords(0), totalRecords(0), cleanedRecords(0) {}
};

// Structure to store location of noise in the data
struct NoiseLocation {
    int record;     // Record index where noise was found
    int parameter;  // Parameter index where noise was found
};

// Global variables for tracking progress across all threads
static std::atomic<int> globalProcessedRecords(0);  // Total records processed
static std::atomic<int> globalCleanedRecords(0);    // Total records cleaned
static EventHandler globalEventHandler;             // Global event handler

//==============================================================================
// Main Data Preprocessor Class
//==============================================================================

class DataPreprocessor {
public:
    // Constants defining the size and structure of the data
    static const int NUM_PARAMETERS = 50;    // Number of parameters per record
    static const int CHUNK_SIZE = 1000;      // Size of data chunks to process
    static const int TOTAL_RECORDS = 1000000;  // Total number of records

private:
    std::vector<double> data;           // Storage for the data being processed
    PreprocessingProgress progress;      // Tracks processing progress
    int threadId;                       // ID of the current thread

    // Calculates median of a vector of values
    double calculateMedian(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        
        if (sorted.size() % 2 == 0) {
            return (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
        }
        return sorted[sorted.size()/2];
    }

    // Cleans data by replacing noise (-1) with calculated medians
    void cleanData() {
        std::vector<double> medians(NUM_PARAMETERS);
        std::vector<double> means(NUM_PARAMETERS);
        std::vector<NoiseLocation> noiseLocations;
        
        // First pass: Calculate medians and means for each parameter
        for (int j = 0; j < NUM_PARAMETERS; j++) {
            std::vector<double> validValues;
            double sum = 0;
            int count = 0;
            
            // Collect valid values and calculate sum
            for (int i = 0; i < CHUNK_SIZE; i++) {
                double value = data[i * NUM_PARAMETERS + j];
                if (value != -1) {
                    validValues.push_back(value);
                    sum += value;
                    count++;
                }
            }
            
            // Calculate median and mean for the parameter
            medians[j] = calculateMedian(validValues);
            means[j] = count > 0 ? sum / count : 0.0;
        }

        // Second pass: Replace noisy data with medians
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < NUM_PARAMETERS; j++) {
                int idx = i * NUM_PARAMETERS + j;
                if (data[idx] == -1) {
                    data[idx] = medians[j];
                    globalCleanedRecords++;
                    noiseLocations.push_back({i, j});
                }
            }
        }

        // Report detected noise locations
        for (const auto& loc : noiseLocations) {
            EventData eventData = {loc.record, loc.parameter, 0};
            globalEventHandler.triggerEvent(PreprocessingEvent::NOISE_DETECTED, eventData);
        }
        
        // Trigger completion event
        globalEventHandler.triggerEvent(PreprocessingEvent::CLEANING_COMPLETED, 
                                      {0, 0, static_cast<int>(globalProcessedRecords)});
    }

public:
    // Constructor initializes the preprocessor for a specific thread
    DataPreprocessor() : threadId(omp_get_thread_num()) {
        progress.totalRecords = TOTAL_RECORDS;
        data.resize(CHUNK_SIZE * NUM_PARAMETERS);
    }

    // Process a chunk of data
    void processChunk() {
        cleanData();
        
        // Update global progress
        int currentProcessed = globalProcessedRecords.fetch_add(CHUNK_SIZE) + CHUNK_SIZE;
        
        // Trigger progress update event
        EventData progressData = {0, 0, currentProcessed};
        globalEventHandler.triggerEvent(PreprocessingEvent::PROGRESS_UPDATE, progressData);
    }

    // Getters and setters for the data
    std::vector<double>& getData() { return data; }
    void setData(const std::vector<double>& newData) { data = newData; }
};

//==============================================================================
// Data Generation Functions
//==============================================================================

// Generates synthetic dataset with controlled noise
std::vector<double> generateFullDataset(int totalRecords, int numParameters) {
    std::vector<double> fullData(totalRecords * numParameters);
    
    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);
    std::uniform_real_distribution<> noise_dis(0, 1);
    
    // Generate data with controlled noise
    for (int i = 0; i < totalRecords; i++) {
        for (int j = 0; j < numParameters; j++) {
            int idx = i * numParameters + j;
            if (noise_dis(gen) < 0.0005) {  // 0.05% chance of noise
                fullData[idx] = -1;          // Noise value
            } else {
                fullData[idx] = dis(gen);    // Regular data value
            }
        }
    }
    return fullData;
}

//==============================================================================
// Main Function
//==============================================================================

int main(int argc, char** argv) {
    // Set up event handling with lambda function for callbacks
    globalEventHandler.setEventCallback(
        [](PreprocessingEvent event, const EventData& data) {
            int thread_id = omp_get_thread_num();
            switch(event) {
                case PreprocessingEvent::NOISE_DETECTED:
                    std::cout << "Thread " << thread_id << ": Noise detected in record " 
                             << data.recordIndex << ", parameter " << data.parameterIndex 
                             << std::endl;
                    break;
                    
                case PreprocessingEvent::PROGRESS_UPDATE:
                    std::cout << "Total records processed: " << data.recordsProcessed 
                             << " (by Thread " << thread_id << ")" << std::endl;
                    break;
                    
                case PreprocessingEvent::CLEANING_COMPLETED:
                    std::cout << "Thread " << thread_id << ": Chunk cleaning completed. " 
                             << "Total progress: " << data.recordsProcessed << std::endl;
                    break;
            }
        }
    );
    
    // Calculate total number of chunks to process
    const int total_chunks = DataPreprocessor::TOTAL_RECORDS / DataPreprocessor::CHUNK_SIZE;
    
    // Start timing the processing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Generating full dataset...\n";
    std::vector<double> fullData = generateFullDataset(DataPreprocessor::TOTAL_RECORDS, 
                                                     DataPreprocessor::NUM_PARAMETERS);
    
    // Process chunks using OpenMP parallel processing
    #pragma omp parallel
    {
        // Create thread-local preprocessor
        DataPreprocessor threadPreprocessor;
        std::vector<double> thread_chunk(DataPreprocessor::CHUNK_SIZE * 
                                       DataPreprocessor::NUM_PARAMETERS);
        
        // Distribute chunks across threads
        #pragma omp for schedule(dynamic)
        for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
            // Calculate starting index for this chunk
            int start_idx = chunk_idx * DataPreprocessor::CHUNK_SIZE * 
                          DataPreprocessor::NUM_PARAMETERS;
            
            // Copy chunk data to thread-local storage
            std::copy(fullData.begin() + start_idx,
                     fullData.begin() + start_idx + 
                     DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                     thread_chunk.begin());
            
            // Process the chunk
            threadPreprocessor.setData(thread_chunk);
            threadPreprocessor.processChunk();
        }
    }

    // Calculate execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();

    // Calculate memory usage
    std::size_t totalMemoryUsage = sizeof(DataPreprocessor) + 
                                  fullData.capacity() * sizeof(double);

    // Print final performance metrics
    std::cout << "\nFinal Performance Metrics:\n"
              << "Total Processing Time: " << processingTime << " seconds\n"
              << "Total Memory Usage: " << totalMemoryUsage << " bytes\n"
              << "Total chunks processed: " << total_chunks << "\n"
              << "Number of threads used: " << omp_get_max_threads() << "\n"
              << "Total records with noise: " << globalCleanedRecords << "\n"
              << "Total records processed: " << globalProcessedRecords << "\n";

    return 0;
}
