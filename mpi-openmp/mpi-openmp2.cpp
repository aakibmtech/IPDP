#include <mpi.h>
#include <omp.h>
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
// Data Structures
//==============================================================================

struct PreprocessingProgress {
    std::atomic<int> processedRecords{0};
    int totalRecords;
    std::atomic<int> cleanedRecords{0};
};

// Global progress tracking
static std::atomic<int> globalProcessedRecords{0};
static std::atomic<int> globalCleanedRecords{0};
static EventHandler globalEventHandler;

//==============================================================================
// Optimized Data Preprocessor Class
//==============================================================================

class DataPreprocessor {
public:
    static const int NUM_PARAMETERS = 50;
    static const int CHUNK_SIZE = 1000;
    static const int TOTAL_RECORDS = 1000000;

private:
    std::vector<double> data;
    std::vector<double> medians;
    int rank;
    int size;
    PreprocessingProgress progress;
    
    // Cache-aligned atomic counter for thread-safe updates
    alignas(64) std::atomic<int> noiseCount{0};

    void calculateMedians() {
        medians.resize(NUM_PARAMETERS);
        std::vector<std::vector<double>> validValues(NUM_PARAMETERS);

        // Collect valid values for each parameter
        for (int i = 0; i < CHUNK_SIZE; i++) {
            for (int j = 0; j < NUM_PARAMETERS; j++) {
                double value = data[i * NUM_PARAMETERS + j];
                if (value != -1) {
                    validValues[j].push_back(value);
                }
            }
        }

        // Calculate median for each parameter
        #pragma omp parallel for
        for (int j = 0; j < NUM_PARAMETERS; j++) {
            if (validValues[j].empty()) {
                medians[j] = 0.0;
                continue;
            }
            
            auto& values = validValues[j];
            std::sort(values.begin(), values.end());
            
            if (values.size() % 2 == 0) {
                medians[j] = (values[values.size()/2 - 1] + values[values.size()/2]) / 2.0;
            } else {
                medians[j] = values[values.size()/2];
            }
        }
    }

    void processRecord(int recordIdx) {
        bool hasNoise = false;
        int baseIdx = recordIdx * NUM_PARAMETERS;

        // Check and clean noise in this record
        for (int j = 0; j < NUM_PARAMETERS; j++) {
            int idx = baseIdx + j;
            if (data[idx] == -1) {
                data[idx] = medians[j];
                hasNoise = true;
                globalCleanedRecords++;
            }
        }

        if (hasNoise) {
            int currentCount = noiseCount.fetch_add(1, std::memory_order_relaxed);
            EventData eventData = {recordIdx, 0, currentCount};
            globalEventHandler.triggerEvent(PreprocessingEvent::NOISE_DETECTED, eventData);
        }
    }

public:
    DataPreprocessor(int _rank, int _size) 
        : rank(_rank), size(_size) {
        progress.totalRecords = TOTAL_RECORDS;
        data.resize(CHUNK_SIZE * NUM_PARAMETERS);
    }

    void processChunkParallel() {
        // Calculate medians first
        calculateMedians();

        // Process records in parallel with dynamic scheduling
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 16) nowait
            for (int i = 0; i < CHUNK_SIZE; i++) {
                processRecord(i);
            }
        }

        // Update global progress
        int records_processed = CHUNK_SIZE;
        globalProcessedRecords += records_processed;
        
        // Gather total progress across all processes
        int local_processed = globalProcessedRecords.load();
        int total_processed;
        MPI_Allreduce(&local_processed, &total_processed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        EventData progressData = {0, 0, total_processed};
        globalEventHandler.triggerEvent(PreprocessingEvent::PROGRESS_UPDATE, progressData);
    }

    std::vector<double>& getData() { return data; }
    void setData(const std::vector<double>& newData) { data = newData; }
};

//==============================================================================
// Main Function with Optimized Implementation
//==============================================================================

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        std::cerr << "Thread multiple support unavailable\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up event handling
    globalEventHandler.setEventCallback(
        [rank](PreprocessingEvent event, const EventData& data) {
            if (rank == 0 && event == PreprocessingEvent::PROGRESS_UPDATE) {
                std::cout << "Total records processed: " << data.recordsProcessed << std::endl;
            }
        }
    );

    const int total_chunks = DataPreprocessor::TOTAL_RECORDS / DataPreprocessor::CHUNK_SIZE;
    const int chunks_per_process = total_chunks / size;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Only root generates full dataset
    std::vector<double> fullData;
    if (rank == 0) {
        fullData.resize(DataPreprocessor::TOTAL_RECORDS * DataPreprocessor::NUM_PARAMETERS);
        
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 99);
            std::uniform_real_distribution<> noise_dis(0, 1);
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < DataPreprocessor::TOTAL_RECORDS; i++) {
                for (int j = 0; j < DataPreprocessor::NUM_PARAMETERS; j++) {
                    int idx = i * DataPreprocessor::NUM_PARAMETERS + j;
                    fullData[idx] = (noise_dis(gen) < 0.05) ? -1 : dis(gen);
                }
            }
        }
    }

    // Process chunks
    DataPreprocessor preprocessor(rank, size);
    std::vector<double> chunk(DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS);

    for (int chunk_idx = 0; chunk_idx < chunks_per_process; chunk_idx++) {
        if (rank == 0) {
            // Distribute chunks
            for (int p = 0; p < size; p++) {
                int base_chunk_idx = chunk_idx * size + p;
                int start_idx = base_chunk_idx * DataPreprocessor::CHUNK_SIZE * 
                              DataPreprocessor::NUM_PARAMETERS;

                if (p == 0) {
                    std::copy(fullData.begin() + start_idx,
                             fullData.begin() + start_idx + chunk.size(),
                             chunk.begin());
                } else {
                    MPI_Send(&fullData[start_idx], chunk.size(), MPI_DOUBLE, 
                            p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(chunk.data(), chunk.size(), MPI_DOUBLE, 
                    0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        preprocessor.setData(chunk);
        preprocessor.processChunkParallel();
    }

    // Gather statistics
    int local_cleaned = globalCleanedRecords.load();
    int total_cleaned;
    MPI_Reduce(&local_cleaned, &total_cleaned, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();

    if (rank == 0) {
        std::vector<double> allTimes(size);
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, allTimes.data(), 1, MPI_DOUBLE, 
                  0, MPI_COMM_WORLD);
        
        double totalTime = *std::max_element(allTimes.begin(), allTimes.end());
        double avgTime = std::accumulate(allTimes.begin(), allTimes.end(), 0.0) / size;

        std::cout << "\nFinal Performance Metrics:\n"
                  << "Total Processing Time: " << totalTime << " seconds\n"
                  << "Average Time per Process: " << avgTime << " seconds\n"
                  << "Total chunks processed: " << total_chunks << "\n"
                  << "Chunks per process: " << chunks_per_process << "\n"
                  << "Threads per process: " << omp_get_max_threads() << "\n"
                  << "Total noisy records cleaned: " << total_cleaned << "\n";
    } else {
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
