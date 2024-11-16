#include <mpi.h>
#include <pthread.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <thread>

//==============================================================================
// Event System
//==============================================================================

enum class PreprocessingEvent {
    NOISE_DETECTED,
    CLEANING_COMPLETED,
    PROGRESS_UPDATE
};

struct EventData {
    int recordIndex;
    int parameterIndex;
    int recordsProcessed;
};

class EventHandler {
private:
    std::function<void(PreprocessingEvent, const EventData&)> eventCallback;
    std::mutex eventMutex;

public:
    void setEventCallback(std::function<void(PreprocessingEvent, const EventData&)> callback) {
        std::lock_guard<std::mutex> lock(eventMutex);
        eventCallback = callback;
    }

    void triggerEvent(PreprocessingEvent event, const EventData& data) {
        std::lock_guard<std::mutex> lock(eventMutex);
        if (eventCallback) {
            eventCallback(event, data);
        }
    }
};

//==============================================================================
// Global Variables
//==============================================================================

static std::atomic<int> globalProcessedRecords(0);
static std::atomic<int> globalCleanedRecords(0);
static EventHandler globalEventHandler;
static std::mutex outputMutex;

//==============================================================================
// Progress Tracking System
//==============================================================================

class ProgressTracker {
private:
    std::mutex progressMutex;
    std::atomic<int> lastReportedPercentage{0};
    const int totalChunks;
    const std::chrono::steady_clock::time_point startTime;

public:
    ProgressTracker(int totalChunks) 
        : totalChunks(totalChunks)
        , startTime(std::chrono::steady_clock::now()) {}

    void updateProgress(int completedChunks) {
        int currentPercentage = (completedChunks * 100) / totalChunks;
        int lastPercentage = lastReportedPercentage.load();
        
        if (currentPercentage >= lastPercentage + 5) {
            if (lastReportedPercentage.compare_exchange_strong(lastPercentage, currentPercentage)) {
                auto currentTime = std::chrono::steady_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                    currentTime - startTime).count();
                
                std::lock_guard<std::mutex> lock(progressMutex);
                std::cout << "\rProcessing: " << currentPercentage << "% complete"
                          << " (Time elapsed: " << elapsedSeconds << "s)" << std::flush;
            }
        }
    }

    void finish() {
        auto endTime = std::chrono::steady_clock::now();
        auto totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(
            endTime - startTime).count();
            
        std::lock_guard<std::mutex> lock(progressMutex);
        std::cout << "\nProcessing completed in " << totalSeconds << " seconds" << std::endl;
    }
};

//==============================================================================
// Data Preprocessor Class
//==============================================================================

class DataPreprocessor {
public:
    static const int NUM_PARAMETERS = 50;
    static const int CHUNK_SIZE = 1000;
    static const int TOTAL_RECORDS = 5000;

private:
    std::vector<double> data;
    int rank;
    int size;

public:
    DataPreprocessor(int _rank, int _size) : rank(_rank), size(_size) {
        data.resize(CHUNK_SIZE * NUM_PARAMETERS);
    }

    std::vector<double>& getData() { return data; }
    void setData(const std::vector<double>& newData) { data = newData; }
};


//==============================================================================
// Thread Task Structure
//==============================================================================

struct ThreadTask {
    int chunkIdx;
    std::vector<double>* fullData;
    int startIdx;
    int endIdx;
};

//==============================================================================
// Thread Pool Implementation
//==============================================================================

class ThreadPool {
private:
    std::vector<pthread_t> threads;
    std::queue<ThreadTask> taskQueue;
    pthread_mutex_t queueMutex;
    pthread_cond_t queueCondition;
    bool stopping;
    int numThreads;

    static void* threadFunction(void* arg) {
        ThreadPool* pool = static_cast<ThreadPool*>(arg);
        while (true) {
            ThreadTask task;
            {
                pthread_mutex_lock(&pool->queueMutex);
                while (pool->taskQueue.empty() && !pool->stopping) {
                    pthread_cond_wait(&pool->queueCondition, &pool->queueMutex);
                }
                if (pool->stopping && pool->taskQueue.empty()) {
                    pthread_mutex_unlock(&pool->queueMutex);
                    break;
                }
                task = pool->taskQueue.front();
                pool->taskQueue.pop();
                pthread_mutex_unlock(&pool->queueMutex);
            }
            pool->processTask(task);
        }
        return nullptr;
    }

    void processTask(const ThreadTask& task) {
        std::vector<double> medians(DataPreprocessor::NUM_PARAMETERS);
        calculateMedians(task.fullData, task.startIdx, task.endIdx, medians);
        cleanData(task.fullData, task.startIdx, task.endIdx, medians);
    }

    void calculateMedians(std::vector<double>* data, int startIdx, int endIdx, 
                         std::vector<double>& medians) {
        for (int j = 0; j < DataPreprocessor::NUM_PARAMETERS; j++) {
            std::vector<double> validValues;
            for (int i = startIdx; i < endIdx; i++) {
                double value = (*data)[i * DataPreprocessor::NUM_PARAMETERS + j];
                if (value != -1) {
                    validValues.push_back(value);
                }
            }
            
            if (!validValues.empty()) {
                std::sort(validValues.begin(), validValues.end());
                medians[j] = validValues.size() % 2 == 0 ?
                    (validValues[validValues.size()/2 - 1] + validValues[validValues.size()/2]) / 2.0 :
                    validValues[validValues.size()/2];
            }
        }
    }

    void cleanData(std::vector<double>* data, int startIdx, int endIdx, 
                  const std::vector<double>& medians) {
        for (int i = startIdx; i < endIdx; i++) {
            for (int j = 0; j < DataPreprocessor::NUM_PARAMETERS; j++) {
                int idx = i * DataPreprocessor::NUM_PARAMETERS + j;
                if ((*data)[idx] == -1) {
                    (*data)[idx] = medians[j];
                    globalCleanedRecords++;
                    EventData eventData = {i, j, 0};
                    globalEventHandler.triggerEvent(PreprocessingEvent::NOISE_DETECTED, eventData);
                }
            }
        }
    }

public:
    ThreadPool(int threads) : numThreads(threads), stopping(false) {
        pthread_mutex_init(&queueMutex, nullptr);
        pthread_cond_init(&queueCondition, nullptr);
        this->threads.resize(threads);
        for (int i = 0; i < threads; i++) {
            pthread_create(&this->threads[i], nullptr, threadFunction, this);
        }
    }

    ~ThreadPool() {
        {
            pthread_mutex_lock(&queueMutex);
            stopping = true;
            pthread_mutex_unlock(&queueMutex);
        }
        pthread_cond_broadcast(&queueCondition);
        
        for (pthread_t& thread : threads) {
            pthread_join(thread, nullptr);
        }
        
        pthread_mutex_destroy(&queueMutex);
        pthread_cond_destroy(&queueCondition);
    }

    void addTask(ThreadTask task) {
        pthread_mutex_lock(&queueMutex);
        taskQueue.push(task);
        pthread_mutex_unlock(&queueMutex);
        pthread_cond_signal(&queueCondition);
    }
};


//==============================================================================
// Data Generation Functions
//==============================================================================

std::vector<double> generateFullDataset(int totalRecords, int numParameters) {
    std::vector<double> fullData(totalRecords * numParameters);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);
    std::uniform_real_distribution<> noise_dis(0, 1);
    
    const int progressStep = totalRecords / 10;
    
    std::cout << "Generating dataset: " << std::endl;
    for (int i = 0; i < totalRecords; i++) {
        if (i % progressStep == 0) {
            std::cout << "\rProgress: " << (i * 100 / totalRecords) << "%" << std::flush;
        }
        
        for (int j = 0; j < numParameters; j++) {
            int idx = i * numParameters + j;
            if (noise_dis(gen) < 0.0005) {
                fullData[idx] = -1;
            } else {
                fullData[idx] = dis(gen);
            }
        }
    }
    std::cout << "\rProgress: 100%" << std::endl;
    return fullData;
}

//==============================================================================
// Main Function
//==============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_threads = std::thread::hardware_concurrency();
    const int total_chunks = DataPreprocessor::TOTAL_RECORDS / DataPreprocessor::CHUNK_SIZE;
    const int chunks_per_process = total_chunks / size;
    
    // Create progress tracker
    ProgressTracker tracker(total_chunks);
    
    // Set up event handling
    globalEventHandler.setEventCallback(
        [rank, &tracker](PreprocessingEvent event, const EventData& data) {
            switch(event) {
                case PreprocessingEvent::NOISE_DETECTED:
                    // Only log significant noise patterns
                    if (data.parameterIndex % 10 == 0) {  // Reduce noise logging frequency
                        std::lock_guard<std::mutex> lock(outputMutex);
                        std::cout << "Process " << rank << ": Noise detected in record " 
                                 << data.recordIndex << ", parameter " 
                                 << data.parameterIndex << std::endl;
                    }
                    break;
                    
                case PreprocessingEvent::PROGRESS_UPDATE:
                    if (rank == 0) {  // Only root process updates progress
                        tracker.updateProgress(data.recordsProcessed / DataPreprocessor::CHUNK_SIZE);
                    }
                    break;
            }
        }
    );

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Generate or receive data
    std::vector<double> fullData;
    if (rank == 0) {
        std::cout << "Generating full dataset...\n";
        fullData = generateFullDataset(DataPreprocessor::TOTAL_RECORDS, 
                                     DataPreprocessor::NUM_PARAMETERS);
    }

    // Create thread pool and preprocessor
    ThreadPool pool(num_threads);
    std::vector<double> chunk(DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS);
    
    // Process chunks
    for (int chunk_idx = 0; chunk_idx < chunks_per_process; chunk_idx++) {
        if (rank == 0) {
            // Distribute chunks to all processes
            for (int p = 0; p < size; p++) {
                int global_chunk_idx = chunk_idx * size + p;
                int start_idx = global_chunk_idx * DataPreprocessor::CHUNK_SIZE * 
                              DataPreprocessor::NUM_PARAMETERS;
                
                std::vector<double> send_chunk(fullData.begin() + start_idx,
                                             fullData.begin() + start_idx + 
                                             DataPreprocessor::CHUNK_SIZE * 
                                             DataPreprocessor::NUM_PARAMETERS);
                
                if (p == 0) {
                    chunk = send_chunk;
                } else {
                    MPI_Send(send_chunk.data(),
                            DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                            MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(chunk.data(),
                    DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Process chunk using thread pool
        int records_per_thread = DataPreprocessor::CHUNK_SIZE / num_threads;
        for (int t = 0; t < num_threads; t++) {
            int start_record = t * records_per_thread;
            int end_record = (t == num_threads - 1) ? 
                            DataPreprocessor::CHUNK_SIZE : 
                            start_record + records_per_thread;
            
            ThreadTask task = {
                chunk_idx,
                &chunk,
                start_record,
                end_record
            };
            pool.addTask(task);
        }
    }

    // Gather final statistics
    int local_cleaned = globalCleanedRecords.load();
    int total_cleaned;
    MPI_Reduce(&local_cleaned, &total_cleaned, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();

    if (rank == 0) {
        tracker.finish();

        std::vector<double> allTimes(size);
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, 
                  allTimes.data(), 1, MPI_DOUBLE, 
                  0, MPI_COMM_WORLD);
        
        double totalTime = *std::max_element(allTimes.begin(), allTimes.end());
        double avgTime = std::accumulate(allTimes.begin(), allTimes.end(), 0.0) / size;

        std::cout << "\nFinal Performance Metrics:\n"
                  << "Total Processing Time: " << totalTime << " seconds\n"
                  << "Average Time per Process: " << avgTime << " seconds\n"
                  << "Total chunks processed: " << total_chunks << "\n"
                  << "Chunks per process: " << chunks_per_process << "\n"
                  << "Threads per process: " << num_threads << "\n"
                  << "Total records with noise: " << globalCleanedRecords << "\n"
                  << "Total records processed: " << globalProcessedRecords << "\n";
    } else {
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
