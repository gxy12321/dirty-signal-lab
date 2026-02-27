// Toy example: rolling mean acceleration (not wired into Python build)
#include <vector>
#include <iostream>

std::vector<double> rolling_mean(const std::vector<double>& x, int window) {
    std::vector<double> out(x.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i];
        if (i >= (size_t)window) sum -= x[i - window];
        if (i >= (size_t)window - 1) out[i] = sum / window;
    }
    return out;
}

int main() {
    std::vector<double> x = {1,2,3,4,5,6,7,8,9,10};
    auto out = rolling_mean(x, 3);
    for (auto v : out) std::cout << v << " ";
    std::cout << std::endl;
}
