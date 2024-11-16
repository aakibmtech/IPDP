// Standard C++ library includes for various functionalities
#include <thread>         // C++ threading support
#include <vector>         // Dynamic array container
#include <random>         // Random number generation
#include <iostream>       // Input/output operations
#include <chrono>        // Time-related functionality
#include <algorithm>      // STL algorithms like sort
#include <numeric>        // Numeric operations
#include <cmath>         // Mathematical functions
#include <functional>     // Function objects and callbacks
#include <mutex>         // Mutual exclusion primitives
#include <atomic>        // Atomic operations
#include <queue>         // Queue container
#include <condition_variable>  // Thread synchronization primitives

//==============================================================================
// Progress Tracking System - Handles real-time progress monitoring
//==============================================================================

class ProgressTracker {
private:
    std::mutex progressMutex;            // Protects progress updates
    std::atomic<int> lastReportedPercentage{0};  // Last reported progress percentage
    const int totalChunks;               // Total number of chunks to process
    const std::chrono::steady_clock::time_point startTime;  // Processing start time

public:
    // Constructor initializes the tracker with total work units
    ProgressTracker(int totalChunks) 
        : totalChunks(totalChunks)
        , startTime(std::chrono::steady_clock::now()) {}

    // Updates and reports progress if significant change occurred
    void updateProgress(int completedChunks) {
        // Calculate current progress percentage
        int currentPercentage = (completedChunks * 100) / totalChunks;
        int lastPercentage = lastReportedPercentage.load();
        
        // Only report if we've progressed by at least 5%
        if (currentPercentage >= lastPercentage + 5) {
            // Atomic compare and exchange to prevent multiple threads from reporting same progress
            if (lastReportedPercentage.compare_exchange_strong(lastPercentage, currentPercentage)) {
                // Calculate elapsed time
                auto currentTime = std::chrono::steady_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                    currentTime - startTime).count();
                
                // Thread-safe progress output
                std::lock_guard<std::mutex> lock(progressMutex);
                std::cout << "\rProcessing: " << currentPercentage << "% complete"
                          << " (Time elapsed: " << elapsedSeconds << "s)" << std::flush;
            }
        }
    }

    // Reports final completion time
    void finish() {
        auto endTime = std::chrono::steady_clock::now();
        auto totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(
            endTime - startTime).count();
            
        std::lock_guard<std::mutex> lock(progressMutex);
        std::cout << "\nProcessing completed in " << totalSeconds << " seconds" << std::endl;
    }
};

//==============================================================================
// Event System - Handles system events and notifications
//==============================================================================

// Different types of preprocessing events that can occur
enum class PreprocessingEvent {
    NOISE_DETECTED,       // When noise is found in data
    CLEANING_COMPLETED,   // When cleaning process completes
    PROGRESS_UPDATE       // For progress reporting
};

// Structure to hold event-related data
struct EventData {
    int recordIndex;      // Which record triggered the event
    int parameterIndex;   // Which parameter in the record
    int recordsProcessed; // How many records processed so far
};

// Handles event registration and triggering
class EventHandler {
private:
    std::function<void(PreprocessingEvent, const EventData&)> eventCallback;  // Event callback function
    std::mutex eventMutex;  // Protects callback registration/triggering

public:
    // Registers a callback function for events
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
// Data Structures - Core data structures for preprocessing
//==============================================================================

// Tracks preprocessing progress across threads
struct PreprocessingProgress {
    std::atomic<int> processedRecords;  // Number of records processed
    int totalRecords;                   // Total records to process
    std::atomic<int> cleanedRecords;    // Number of records cleaned
    
    // Constructor initializes counters to zero
    PreprocessingProgress() : processedRecords(0), totalRecords(0), cleanedRecords(0) {}
};

// Stores location of noise in data
struct NoiseLocation {
    int record;     // Record index
    int parameter;  // Parameter index
};

// Represents a task for thread pool
struct ThreadTask {
    int chunkIdx;              // Index of the chunk to process
    std::vector<double>* fullData;  // Pointer to full dataset
    int startIdx;             // Start index of chunk
    int endIdx;               // End index of chunk
};

// Thread pool implementation for parallel processing
class ThreadPool {
private:
    std::vector<std::thread> threads;      // Worker threads
    std::queue<ThreadTask> taskQueue;      // Task queue
    std::mutex queueMutex;                // Protects task queue
    std::condition_variable queueCondition; // For thread synchronization
    bool stopping;                        // Signals threads to stop
    int numThreads;                       // Number of worker threads

    // Main worker thread function
    void workerThread() {
        while (true) {
            ThreadTask task;
            {
                // Wait for task or stop signal
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCondition.wait(lock, [this] { 
                    return !taskQueue.empty() || stopping; 
                });
                
                // Check if we should stop
                if (stopping && taskQueue.empty()) {
                    break;
                }
                
                // Get next task
                task = taskQueue.front();
                taskQueue.pop();
            }
            // Process the task
            processTask(task);
        }
    }

    // Task processing function (implemented later)
    void processTask(const ThreadTask& task);

public:
    // Constructor creates worker threads
    ThreadPool(int threads) : numThreads(threads), stopping(false) {
        for (int i = 0; i < threads; i++) {
            this->threads.emplace_back(&ThreadPool::workerThread, this);
        }
    }

    // Destructor ensures clean shutdown
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stopping = true;
        }
        queueCondition.notify_all();
        
        // Wait for all threads to complete
        for (std::thread& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    // Adds a new task to the queue
    void addTask(ThreadTask task) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            taskQueue.push(task);
        }
        queueCondition.notify_one();
    }
};

// Global variables for cross-thread coordination
static std::atomic<int> globalProcessedRecords(0);  // Total records processed
static std::atomic<int> globalCleanedRecords(0);    // Total records cleaned
static EventHandler globalEventHandler;             // Global event handler
static std::mutex outputMutex;                     // Protects console output

//==============================================================================
// Main Data Preprocessor Class - Core processing logic
//==============================================================================

class DataPreprocessor {
public:
    static const int NUM_PARAMETERS = 50;     // Parameters per record
    static const int CHUNK_SIZE = 1000;       // Records per chunk
    static const int TOTAL_RECORDS = 1000000; // Total records to process

private:
    std::vector<double> data;     // Stores chunk data
    PreprocessingProgress progress;  // Tracks progress
    int threadId;                   // Thread identifier

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

    // Cleans data by replacing noise with medians
    void cleanData() {
        std::vector<double> medians(NUM_PARAMETERS);
        std::vector<double> means(NUM_PARAMETERS);
        std::vector<NoiseLocation> noiseLocations;
        
        // First pass: Calculate medians and means
        for (int j = 0; j < NUM_PARAMETERS; j++) {
            std::vector<double> validValues;
            double sum = 0;
            int count = 0;
            
            // Collect valid values for parameter j
            for (int i = 0; i < data.size() / NUM_PARAMETERS; i++) {
                double value = data[i * NUM_PARAMETERS + j];
                if (value != -1) {  // -1 indicates noise
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
        for (int i = 0; i < data.size() / NUM_PARAMETERS; i++) {
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
    }

public:
    // Constructor initializes preprocessor instance
    DataPreprocessor() : threadId(0) {
        progress.totalRecords = TOTAL_RECORDS;
        data.resize(CHUNK_SIZE * NUM_PARAMETERS);
    }

    // Processes a chunk of data
    void processChunk() {
        cleanData();
        
        // Update global progress atomically
        int currentProcessed = globalProcessedRecords.fetch_add(CHUNK_SIZE) + CHUNK_SIZE;
        
        // Report progress
        EventData progressData = {0, 0, currentProcessed};
        globalEventHandler.triggerEvent(PreprocessingEvent::PROGRESS_UPDATE, progressData);
    }

    // Data access methods
    std::vector<double>& getData() { return data; }
    void setData(const std::vector<double>& newData) { data = newData; }
};

// Implementation of ThreadPool's processTask method
void ThreadPool::processTask(const ThreadTask& task) {
    // Create preprocessor instance for this thread
    DataPreprocessor threadPreprocessor;
    
    // Extract chunk from full dataset
    std::vector<double> thread_chunk(
        task.fullData->begin() + task.startIdx * DataPreprocessor::NUM_PARAMETERS,
        task.fullData->begin() + task.endIdx * DataPreprocessor::NUM_PARAMETERS);
            
    // Process the chunk
    threadPreprocessor.setData(thread_chunk);
    threadPreprocessor.processChunk();
}

//==============================================================================
// Data Generation Functions - Creates test dataset
//==============================================================================

// Generates synthetic dataset with controlled noise
std::vector<double> generateFullDataset(int totalRecords, int numParameters) {
    std::vector<double> fullData(totalRecords * numParameters);
    
    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);
    std::uniform_real_distribution<> noise_dis(0, 1);
    
    const int progressStep = totalRecords / 10;  // Report every 10%
    
    std::cout << "Generating dataset: " << std::endl;
    for (int i = 0; i < totalRecords; i++) {
        // Progress reporting
        if (i % progressStep == 0) {
            std::cout << "\rProgress: " << (i * 100 / totalRecords) << "%" << std::flush;
        }
        
        // Generate data with occasional noise
        for (int j = 0; j < numParameters; j++) {
            int idx = i * numParameters + j;
            if (noise_dis(gen) < 0.0005) {  // 0.05% chance of noise
                fullData[idx] = -1;  // Noise value
            } else {
                fullData[idx] = dis(gen);  // Normal value
            }
        }
    }
    std::cout << "\rProgress: 100%" << std::endl;
    return fullData;
}

//==============================================================================
// Main Function - Program entry point
//==============================================================================

//==============================================================================
// Main Function
//==============================================================================

int main(int argc, char** argv) {
    // Get the number of hardware threads available on the system
    // This helps in optimal thread allocation based on the CPU
    int num_threads = std::thread::hardware_concurrency();
    
    // Calculate the total number of data chunks we need to process
    // by dividing total records by the size of each chunk
    const int total_chunks = DataPreprocessor::TOTAL_RECORDS / DataPreprocessor::CHUNK_SIZE;
    
    // Record the start time for performance measurement
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Generate the complete dataset with progress indication
    // This creates our initial data that will be processed
    std::vector<double> fullData = generateFullDataset(DataPreprocessor::TOTAL_RECORDS, 
                                                     DataPreprocessor::NUM_PARAMETERS);
    
    // Initialize progress tracking system
    // This will help monitor the progress of our data processing
    ProgressTracker tracker(total_chunks);
    
    // Set up the event handling system with a lambda function
    // This defines how different events will be handled during processing
    globalEventHandler.setEventCallback(
        [&tracker](PreprocessingEvent event, const EventData& data) {
            switch(event) {
                case PreprocessingEvent::NOISE_DETECTED:
                    // Only log every 10th noise detection to prevent flooding the output
                    if (data.parameterIndex % 10 == 0) {
                        // Use mutex to prevent output interleaving between threads
                        std::lock_guard<std::mutex> lock(outputMutex);
                        std::cout << "Noise detected in record " 
                                 << data.recordIndex << ", parameter " 
                                 << data.parameterIndex << std::endl;
                    }
                    break;
                    
                case PreprocessingEvent::PROGRESS_UPDATE:
                    // Update the progress tracker with the number of chunks processed
                    tracker.updateProgress(data.recordsProcessed / DataPreprocessor::CHUNK_SIZE);
                    break;
            }
        }
    );
    
    // Create and execute thread pool
    {   // Create a new scope for the ThreadPool to control its lifetime
        ThreadPool pool(num_threads);  // Initialize thread pool with available hardware threads
        
        // Distribute work among threads by creating tasks for each chunk
        for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
            // Calculate the start and end records for this chunk
            int start_record = chunk_idx * DataPreprocessor::CHUNK_SIZE;
            int end_record = start_record + DataPreprocessor::CHUNK_SIZE;
            
            // Add the task to the thread pool's queue
            pool.addTask({chunk_idx, &fullData, start_record, end_record});
        }
        
    }  // ThreadPool destructor is called here, ensuring all tasks complete
    
    // Mark processing as complete in the tracker
    tracker.finish();
    
    // Calculate and display final performance metrics
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate total processing time
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();
    // Calculate total memory usage
    std::size_t totalMemoryUsage = sizeof(DataPreprocessor) + 
                                  fullData.capacity() * sizeof(double);

    // Display final performance statistics
    std::cout << "\nFinal Performance Metrics:\n"
              << "Total Processing Time: " << processingTime << " seconds\n"
              << "Total Memory Usage: " << totalMemoryUsage << " bytes\n"
              << "Total chunks processed: " << total_chunks << "\n"
              << "Number of threads used: " << num_threads << "\n"
              << "Total records with noise: " << globalCleanedRecords << "\n"
              << "Total records processed: " << globalProcessedRecords << "\n";

    return 0;
}
