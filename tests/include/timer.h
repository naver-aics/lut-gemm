#include <omp.h>
#include <vector>
#include <algorithm>

class timer {
public:
    std::vector<double> arr;
    bool sort_flag = false;
    double s;
    void start(){
        s = omp_get_wtime();
    }
    double end(){
        double l = (omp_get_wtime() - s) * 1000.0;
        arr.push_back(l);
        sort_flag = false;
        return l;
    }

    double mean(){
        double sum=0;
        for(auto it : arr)
            sum += it;
        return sum/arr.size();
    }

    void sort(){
        if(sort_flag) return;
        std::sort(arr.begin(), arr.end());
        sort_flag = true;
    }

    
    double pile(float p){
        sort();
        int idx = (arr.size() - 1) * p;
        return arr[idx];
    }

    double max(){
        sort();
        return arr[arr.size() - 1];
    }
    double min(){
        sort();
        return arr[0];
    }

    timer(/* args */){}
    ~timer(){}
};

