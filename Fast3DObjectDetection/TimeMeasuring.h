#pragma once
#include <unordered_map>
#include <chrono>
class TimeMeasuring
{
private:
	bool started = false;
	std::chrono::steady_clock::time_point start;
	std::unordered_map<std::string, std::chrono::steady_clock::time_point> breakpoints;
public:
	TimeMeasuring(bool start = false) {
		this->breakpoints = std::unordered_map<std::string, std::chrono::steady_clock::time_point>();
		if (start)
		{
			this->startMeasuring();
		}
	}

	bool startMeasuring() {
		if (!this->started)
		{
			this->start = std::chrono::steady_clock::now();
			this->started = true;
			return true;
		}
		return false;
	}

	bool insertBreakpoint(std::string name) {
		if (this->breakpoints.count(name) == 0)
		{
			this->breakpoints[name] = std::chrono::steady_clock::now();
			return true;
		}
		return false;
	}

	long long int getTimeFromBeginning(bool microSeconds = false) {
		if (!this->started)
		{
			return -1;
		}
		if (microSeconds)
		{
			return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - this->start).count();
		}
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - this->start).count();
	}

	long long int getTimeFromBreakpoint(std::string breakpoint, bool microSeconds = false) {
		if (!this->started || this->breakpoints.count(breakpoint) == 0)
		{
			return -1;
		}
		if (microSeconds)
		{
			return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - this->breakpoints[breakpoint]).count();
		}
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - this->breakpoints[breakpoint]).count();
	}
};

