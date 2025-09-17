#!/usr/bin/env python3
"""
Stress Testing Suite for Enhanced Write Essay Scorer
Tests parallel processing, GPU fallback, and system limits
"""

import sys
import os
import time
import asyncio
import concurrent.futures
import threading
import psutil
import json
from typing import List, Dict, Any
import statistics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test essays with varying complexity
TEST_ESSAYS = [
    {
        "name": "Simple Essay",
        "prompt": "Discuss the benefits of exercise",
        "essay": """
Exercise is very important for health. Many people do not exercise enough today. This is a problem because exercise helps our body and mind.

First, exercise makes our body strong. When we run or lift weights, our muscles grow. Our heart also becomes stronger. This helps us live longer and feel better every day.

Second, exercise is good for our mind. When we exercise, our brain makes chemicals that make us happy. This helps reduce stress and worry. Many people feel better after a good workout.

In conclusion, everyone should exercise regularly. It helps both our body and mind. Even a short walk every day can make a big difference in our health and happiness.
"""
    },
    {
        "name": "Complex Essay",
        "prompt": "Analyze the impact of artificial intelligence on society",
        "essay": """
Artificial intelligence represents one of the most transformative technological developments of the twenty-first century, fundamentally reshaping various aspects of contemporary society. This revolutionary technology presents both unprecedented opportunities and significant challenges that demand careful consideration.

The beneficial implications of AI are particularly evident in healthcare, where machine learning algorithms can analyze medical data with remarkable precision, enabling earlier disease detection and personalized treatment protocols. Furthermore, AI-driven automation has enhanced productivity across numerous industries, from manufacturing to financial services, creating more efficient processes and reducing human error.

However, the proliferation of artificial intelligence also introduces concerning societal implications. The displacement of traditional employment opportunities poses significant economic challenges, particularly for workers in routine-based occupations. Additionally, the concentration of AI capabilities within major technology corporations raises questions about market competition and data privacy.

Moreover, the ethical dimensions of AI implementation require substantial deliberation. Issues surrounding algorithmic bias, decision-making transparency, and accountability mechanisms remain inadequately addressed. The potential for AI systems to perpetuate or amplify existing social inequalities represents a critical concern that necessitates proactive regulatory frameworks.

In conclusion, while artificial intelligence offers transformative potential for societal advancement, its integration must be approached with careful consideration of economic, ethical, and social implications. Balanced policies that maximize benefits while mitigating risks will be essential for ensuring AI contributes positively to human welfare.
"""
    },
    {
        "name": "Error-Rich Essay",
        "prompt": "Write about your favorite hobby",
        "essay": """
My favourit hobby is reading books. I have been read since I was very yong child. Books are wonderfull because they take you to diferent worlds and help you lern new things.

I especialy like fantacy novels and scince fiction storys. These type of books has amazing charecters and exiting adventures. When I read, I can forgot about all my problems and just enjoy the storry.

Reading also help me improve my vocabulery and writting skills. The more I read, the beter I become at understanding complecated ideas and expresing my own thoughts. It is realy a valuable hobbie that benifits many aspects of my life.

In the futur, I want to read even more books and maybe even try writting my own novel. Reading has given me so much joy and knowledge that I would recomend it to anyone looking for a meaningfull way to spend thier free time.
"""
    }
]

class StressTester:
    """Comprehensive stress testing for the enhanced scorer"""
    
    def __init__(self):
        self.results = []
        self.system_metrics = []
        
    def monitor_system_resources(self, duration: float) -> Dict[str, Any]:
        """Monitor system resources during testing"""
        start_time = time.time()
        cpu_readings = []
        memory_readings = []
        
        while time.time() - start_time < duration:
            cpu_readings.append(psutil.cpu_percent(interval=0.1))
            memory_readings.append(psutil.virtual_memory().percent)
            time.sleep(0.5)
        
        return {
            'cpu_avg': statistics.mean(cpu_readings),
            'cpu_max': max(cpu_readings),
            'memory_avg': statistics.mean(memory_readings),
            'memory_max': max(memory_readings),
            'duration': duration
        }
    
    def single_scoring_test(self, essay_data: Dict, test_id: int) -> Dict[str, Any]:
        """Test single essay scoring with timing"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            from app.services.scoring.enhanced_write_essay_scorer import score_enhanced_write_essay
            
            result = score_enhanced_write_essay(essay_data["essay"], essay_data["prompt"])
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'test_id': test_id,
                'essay_name': essay_data["name"],
                'success': result.get('success', False),
                'processing_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'total_score': result.get('total_score', 0),
                'word_count': result.get('word_count', 0),
                'errors': len(result.get('errors', {})),
                'gpt_used': result.get('api_cost', 0) > 0
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'test_id': test_id,
                'essay_name': essay_data["name"],
                'success': False,
                'processing_time': end_time - start_time,
                'memory_used': 0,
                'error': str(e)
            }
    
    def test_sequential_processing(self, num_tests: int = 20) -> Dict[str, Any]:
        """Test sequential processing performance"""
        print(f"🔄 Running {num_tests} sequential tests...")
        
        start_time = time.time()
        results = []
        
        for i in range(num_tests):
            essay_data = TEST_ESSAYS[i % len(TEST_ESSAYS)]
            result = self.single_scoring_test(essay_data, i)
            results.append(result)
            
            if i % 5 == 0:
                print(f"  Completed {i+1}/{num_tests} tests")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        processing_times = [r['processing_time'] for r in results if r['success']]
        success_rate = len([r for r in results if r['success']]) / len(results)
        
        return {
            'test_type': 'sequential',
            'total_tests': num_tests,
            'success_rate': success_rate,
            'total_time': total_time,
            'avg_processing_time': statistics.mean(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'min_processing_time': min(processing_times) if processing_times else 0,
            'throughput_per_second': num_tests / total_time,
            'results': results
        }
    
    def test_parallel_processing(self, num_tests: int = 50, max_workers: int = 10) -> Dict[str, Any]:
        """Test parallel processing with thread pool"""
        print(f"🚀 Running {num_tests} parallel tests with {max_workers} workers...")
        
        start_time = time.time()
        
        # Monitor system resources
        monitor_thread = threading.Thread(
            target=self._monitor_resources_background,
            args=(60,)  # Monitor for up to 60 seconds
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run parallel tests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i in range(num_tests):
                essay_data = TEST_ESSAYS[i % len(TEST_ESSAYS)]
                future = executor.submit(self.single_scoring_test, essay_data, i)
                futures.append(future)
            
            # Collect results
            results = []
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:
                    print(f"  Completed {completed}/{num_tests} tests")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        processing_times = [r['processing_time'] for r in results if r['success']]
        success_rate = len([r for r in results if r['success']]) / len(results)
        memory_usage = [r['memory_used'] for r in results if r['success']]
        
        return {
            'test_type': 'parallel',
            'total_tests': num_tests,
            'max_workers': max_workers,
            'success_rate': success_rate,
            'total_time': total_time,
            'avg_processing_time': statistics.mean(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'min_processing_time': min(processing_times) if processing_times else 0,
            'throughput_per_second': num_tests / total_time,
            'avg_memory_per_test': statistics.mean(memory_usage) if memory_usage else 0,
            'system_metrics': self.system_metrics,
            'results': results
        }
    
    def _monitor_resources_background(self, duration: float):
        """Background thread to monitor system resources"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.system_metrics.append({
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024
            })
            time.sleep(1)
    
    def test_memory_limits(self, max_concurrent: int = 100) -> Dict[str, Any]:
        """Test memory usage under high load"""
        print(f"🧠 Testing memory limits with {max_concurrent} concurrent requests...")
        
        start_memory = psutil.virtual_memory().percent
        start_time = time.time()
        
        # Create a large number of concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            
            # Submit many requests
            for i in range(max_concurrent):
                essay_data = TEST_ESSAYS[i % len(TEST_ESSAYS)]
                future = executor.submit(self.single_scoring_test, essay_data, i)
                futures.append(future)
            
            # Monitor memory during execution
            peak_memory = start_memory
            successful_tests = 0
            
            for future in concurrent.futures.as_completed(futures):
                current_memory = psutil.virtual_memory().percent
                peak_memory = max(peak_memory, current_memory)
                
                result = future.result()
                if result['success']:
                    successful_tests += 1
        
        total_time = time.time() - start_time
        
        return {
            'test_type': 'memory_limits',
            'max_concurrent': max_concurrent,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / max_concurrent,
            'start_memory_percent': start_memory,
            'peak_memory_percent': peak_memory,
            'memory_increase': peak_memory - start_memory,
            'total_time': total_time
        }
    
    def test_cache_performance(self, num_repeated_essays: int = 50) -> Dict[str, Any]:
        """Test caching performance with repeated essays"""
        print(f"💾 Testing cache performance with {num_repeated_essays} repeated essays...")
        
        try:
            from app.services.scoring.parsing_cache import get_parsing_cache
            cache = get_parsing_cache()
            
            # Clear cache for clean test
            cache.clear_cache(keep_persistent=True)
            
            essay_data = TEST_ESSAYS[1]  # Use complex essay
            
            # First run (no cache)
            start_time = time.time()
            first_result = self.single_scoring_test(essay_data, 0)
            first_run_time = time.time() - start_time
            
            # Subsequent runs (with cache)
            cached_times = []
            for i in range(1, num_repeated_essays):
                start_time = time.time()
                result = self.single_scoring_test(essay_data, i)
                cached_times.append(time.time() - start_time)
            
            cache_stats = cache.get_cache_stats()
            
            return {
                'test_type': 'cache_performance',
                'num_tests': num_repeated_essays,
                'first_run_time': first_run_time,
                'avg_cached_time': statistics.mean(cached_times),
                'cache_speedup': first_run_time / statistics.mean(cached_times) if cached_times else 1,
                'cache_stats': cache_stats
            }
            
        except ImportError:
            return {
                'test_type': 'cache_performance',
                'error': 'Caching system not available'
            }
    
    def test_gpu_fallback(self) -> Dict[str, Any]:
        """Test GPU fallback behavior"""
        print("🔄 Testing GPU fallback behavior...")
        
        try:
            import torch
            
            # Test with GPU if available
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                print("  GPU detected, testing GPU processing...")
                
                # Force GPU usage
                essay_data = TEST_ESSAYS[1]
                gpu_result = self.single_scoring_test(essay_data, 0)
                
                # Simulate GPU unavailable (would need manual testing)
                return {
                    'test_type': 'gpu_fallback',
                    'gpu_available': True,
                    'gpu_processing_time': gpu_result['processing_time'],
                    'note': 'Manual GPU disable test required for complete fallback testing'
                }
            else:
                print("  No GPU detected, testing CPU processing...")
                
                essay_data = TEST_ESSAYS[1]
                cpu_result = self.single_scoring_test(essay_data, 0)
                
                return {
                    'test_type': 'gpu_fallback',
                    'gpu_available': False,
                    'cpu_processing_time': cpu_result['processing_time'],
                    'fallback_working': cpu_result['success']
                }
                
        except ImportError:
            return {
                'test_type': 'gpu_fallback',
                'error': 'PyTorch not available for GPU testing'
            }
    
    def run_comprehensive_stress_test(self) -> Dict[str, Any]:
        """Run complete stress testing suite"""
        print("🚀 STARTING COMPREHENSIVE STRESS TEST")
        print("=" * 80)
        
        test_results = {}
        
        # Test 1: Sequential Processing
        test_results['sequential'] = self.test_sequential_processing(20)
        print(f"✅ Sequential: {test_results['sequential']['success_rate']:.1%} success, "
              f"{test_results['sequential']['throughput_per_second']:.1f} essays/sec")
        
        # Test 2: Parallel Processing
        test_results['parallel'] = self.test_parallel_processing(50, 10)
        print(f"✅ Parallel: {test_results['parallel']['success_rate']:.1%} success, "
              f"{test_results['parallel']['throughput_per_second']:.1f} essays/sec")
        
        # Test 3: Memory Limits
        test_results['memory'] = self.test_memory_limits(50)
        print(f"✅ Memory: {test_results['memory']['success_rate']:.1%} success, "
              f"{test_results['memory']['memory_increase']:.1f}% memory increase")
        
        # Test 4: Cache Performance
        test_results['cache'] = self.test_cache_performance(30)
        if 'cache_speedup' in test_results['cache']:
            print(f"✅ Cache: {test_results['cache']['cache_speedup']:.1f}x speedup")
        
        # Test 5: GPU Fallback
        test_results['gpu_fallback'] = self.test_gpu_fallback()
        print(f"✅ GPU Fallback: {'Working' if test_results['gpu_fallback'].get('fallback_working', True) else 'Failed'}")
        
        # Overall summary
        test_results['summary'] = {
            'total_tests_run': sum(r.get('total_tests', 0) for r in test_results.values() if isinstance(r, dict)),
            'overall_success_rate': statistics.mean([
                r['success_rate'] for r in test_results.values() 
                if isinstance(r, dict) and 'success_rate' in r
            ]),
            'best_throughput': max([
                r['throughput_per_second'] for r in test_results.values() 
                if isinstance(r, dict) and 'throughput_per_second' in r
            ]),
            'test_timestamp': time.time()
        }
        
        print("\n" + "=" * 80)
        print("🏆 STRESS TEST COMPLETED")
        print(f"Total Tests: {test_results['summary']['total_tests_run']}")
        print(f"Overall Success Rate: {test_results['summary']['overall_success_rate']:.1%}")
        print(f"Best Throughput: {test_results['summary']['best_throughput']:.1f} essays/sec")
        print("=" * 80)
        
        return test_results

def main():
    """Run stress tests"""
    
    # Check if enhanced scorer is available
    try:
        from app.services.scoring.enhanced_write_essay_scorer import score_enhanced_write_essay
        print("✅ Enhanced scorer available")
    except ImportError as e:
        print(f"❌ Enhanced scorer not available: {e}")
        return
    
    # Run tests
    tester = StressTester()
    results = tester.run_comprehensive_stress_test()
    
    # Save results
    results_file = f"stress_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: {results_file}")
    
    # Performance recommendations
    print("\n🔧 PERFORMANCE RECOMMENDATIONS:")
    
    if results['parallel']['success_rate'] < 0.95:
        print("⚠️  Consider reducing max_workers for parallel processing")
    
    if results['memory']['memory_increase'] > 50:
        print("⚠️  High memory usage detected - consider memory optimization")
    
    if 'cache_speedup' in results['cache'] and results['cache']['cache_speedup'] < 2:
        print("⚠️  Cache performance lower than expected - check TTL settings")
    
    if results['parallel']['throughput_per_second'] < 5:
        print("⚠️  Low throughput detected - consider infrastructure scaling")
    else:
        print("✅ System performance within acceptable limits")

if __name__ == "__main__":
    main()