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
    std::atomic<int> processedRecords;
    int totalRecords;
    std::atomic<int> cleanedRecords;
    
    PreprocessingProgress() : processedRecords(0), totalRecords(0), cleanedRecords(0) {}
};

struct NoiseLocation {
    int record;
    int parameter;
};

// Global progress tracking per process
static std::atomic<int> globalProcessedRecords(0);
static std::atomic<int> globalCleanedRecords(0);
static EventHandler globalEventHandler;

//==============================================================================
// Main Data Preprocessor Class
//==============================================================================

class DataPreprocessor {
public:
    static const int NUM_PARAMETERS = 50;
    static const int CHUNK_SIZE = 1000;
    static const int TOTAL_RECORDS = 10000000;

private:
    std::vector<double> data;
    int rank;
    int size;
    int threadId;
    PreprocessingProgress progress;

    double calculateMedian(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        
        if (sorted.size() % 2 == 0) {
            return (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
        }
        return sorted[sorted.size()/2];
    }

    void cleanData() {
        std::vector<double> medians(NUM_PARAMETERS);
        std::vector<double> means(NUM_PARAMETERS);
        std::vector<NoiseLocation> noiseLocations;
        
        // First pass: Calculate medians and means for each parameter
        for (int j = 0; j < NUM_PARAMETERS; j++) {
            std::vector<double> validValues;
            double sum = 0;
            int count = 0;
            
            for (int i = 0; i < data.size() / NUM_PARAMETERS; i++) {
                double value = data[i * NUM_PARAMETERS + j];
                if (value != -1) {
                    validValues.push_back(value);
                    sum += value;
                    count++;
                }
            }
            
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

        // Report noise locations
        for (const auto& loc : noiseLocations) {
            EventData eventData = {loc.record, loc.parameter, 0};
            globalEventHandler.triggerEvent(PreprocessingEvent::NOISE_DETECTED, eventData);
        }
    }

public:
    DataPreprocessor(int _rank, int _size) 
        : rank(_rank), size(_size), threadId(omp_get_thread_num()) {
        progress.totalRecords = TOTAL_RECORDS;
    }

    void processChunk() {
        cleanData();
        
        // Update local progress with actual processed records
        int records_processed = data.size() / NUM_PARAMETERS;
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
// Data Generation Functions
//==============================================================================

std::vector<double> generateFullDataset(int totalRecords, int numParameters) {
    std::vector<double> fullData(totalRecords * numParameters);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);
    std::uniform_real_distribution<> noise_dis(0, 1);
    
    for (int i = 0; i < totalRecords; i++) {
        for (int j = 0; j < numParameters; j++) {
            int idx = i * numParameters + j;
            if (noise_dis(gen) < 0.05) {
                fullData[idx] = -1;
            } else {
                fullData[idx] = dis(gen);
            }
        }
    }
    return fullData;
}

//==============================================================================
// Main Function
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
            int thread_id = omp_get_thread_num();
            switch(event) {
                case PreprocessingEvent::NOISE_DETECTED:
                    std::cout << "Process " << rank << ", Thread " << thread_id 
                             << ": Noise detected in record " << data.recordIndex 
                             << ", parameter " << data.parameterIndex << std::endl;
                    break;
                    
                case PreprocessingEvent::PROGRESS_UPDATE:
                    if (rank == 0 && thread_id == 0) {  // Only print from main process, main thread
                        std::cout << "Total records processed across all processes: " 
                                 << data.recordsProcessed << std::endl;
                    }
                    break;
                    
                case PreprocessingEvent::CLEANING_COMPLETED:
                    std::cout << "Process " << rank << ", Thread " << thread_id 
                             << ": Cleaning completed" << std::endl;
                    break;
            }
        }
    );
    
    const int total_chunks = DataPreprocessor::TOTAL_RECORDS / DataPreprocessor::CHUNK_SIZE;
    const int chunks_per_process = total_chunks / size;
    
    
    std::vector<double> fullData;
    std::vector<double> process_chunk(DataPreprocessor::CHUNK_SIZE * 
                                    DataPreprocessor::NUM_PARAMETERS);
    
    if (rank == 0) {
                std::cout << "Generating full dataset...\n";
                fullData = generateFullDataset(DataPreprocessor::TOTAL_RECORDS, 
                                            DataPreprocessor::NUM_PARAMETERS);
            }
    
    auto startTime = std::chrono::high_resolution_clock::now();

    // Process multiple chunks
    for (int chunk_idx = 0; chunk_idx < chunks_per_process; chunk_idx++) {
        if (rank == 0) {
            
            // Distribute chunks to all processes
            for (int p = 0; p < size; p++) {
                int base_chunk_idx = chunk_idx * size + p;
                int start_idx = base_chunk_idx * DataPreprocessor::CHUNK_SIZE * 
                              DataPreprocessor::NUM_PARAMETERS;
                
                std::vector<double> send_chunk(fullData.begin() + start_idx,
                                             fullData.begin() + start_idx + 
                                             DataPreprocessor::CHUNK_SIZE * 
                                             DataPreprocessor::NUM_PARAMETERS);
                
                if (p == 0) {
                    process_chunk = send_chunk;
                } else {
                    MPI_Send(send_chunk.data(),
                           DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                           MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            MPI_Recv(process_chunk.data(),
                    DataPreprocessor::CHUNK_SIZE * DataPreprocessor::NUM_PARAMETERS,
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Process chunk using OpenMP threads
        #pragma omp parallel
        {
            DataPreprocessor threadPreprocessor(rank, size);
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();
            
            // Calculate portion for each thread
            int records_per_thread = DataPreprocessor::CHUNK_SIZE / num_threads;
            int start_record = thread_id * records_per_thread;
            int end_record = (thread_id == num_threads - 1) ? 
                            DataPreprocessor::CHUNK_SIZE : 
                            start_record + records_per_thread;
            
            // Create thread's portion of the chunk
            std::vector<double> thread_chunk((end_record - start_record) * 
                                           DataPreprocessor::NUM_PARAMETERS);
            
            // Copy thread's portion of data
            std::copy(process_chunk.begin() + start_record * DataPreprocessor::NUM_PARAMETERS,
                     process_chunk.begin() + end_record * DataPreprocessor::NUM_PARAMETERS,
                     thread_chunk.begin());
            
            threadPreprocessor.setData(thread_chunk);
            threadPreprocessor.processChunk();
        }
    }

    // Gather final statistics
    int local_cleaned = globalCleanedRecords.load();
    int total_cleaned;
    MPI_Reduce(&local_cleaned, &total_cleaned, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(endTime - startTime).count();

    std::size_t localMemoryUsage = sizeof(DataPreprocessor) * omp_get_max_threads() + 
                                  process_chunk.capacity() * sizeof(double);
    if (rank == 0) {
        localMemoryUsage += fullData.capacity() * sizeof(double);
    }

    if (rank == 0) {
        std::vector<double> allTimes(size);
        std::vector<std::size_t> allMemory(size);

        MPI_Gather(&processingTime, 1, MPI_DOUBLE, 
                  allTimes.data(), 1, MPI_DOUBLE, 
                  0, MPI_COMM_WORLD);

        MPI_Gather(&localMemoryUsage, 1, MPI_DOUBLE, 
                  allMemory.data(), 1, MPI_DOUBLE, 
                  0, MPI_COMM_WORLD);
        
        double totalTime = *std::max_element(allTimes.begin(), allTimes.end());
        double avgTime = std::accumulate(allTimes.begin(), allTimes.end(), 0.0) / size;
        std::size_t totalMemory = std::accumulate(allMemory.begin(), allMemory.end(), 0ULL);

        std::cout << "\nFinal Performance Metrics:\n"
                  << "Total Processing Time: " << totalTime << " seconds\n"
                  << "Average Time per Process: " << avgTime << " seconds\n"
                  << "Total Memory Usage: " << totalMemory << " bytes\n"
                  << "Total chunks processed: " << total_chunks << "\n"
                  << "Chunks per process: " << chunks_per_process << "\n"
                  << "Threads per process: " << omp_get_max_threads() << "\n"
                  << "Total noisy records cleaned: " << total_cleaned << "\n";
    } else {
        MPI_Gather(&processingTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&localMemoryUsage, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
