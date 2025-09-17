#!/usr/bin/env python3
"""
Configuration Management API Endpoints
Handles persistent calibration, academic whitelist, and model configuration
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response validation
class CalibrationData(BaseModel):
    raw_scores: Dict[str, float]
    expected_score: float

class CalibrationRequest(BaseModel):
    calibration_data: List[CalibrationData]
    description: Optional[str] = None

class AcademicWhitelistRequest(BaseModel):
    terms: List[str]
    action: str  # "add" or "remove"

class ModelConfigRequest(BaseModel):
    config: Dict[str, Any]

class ConfigExportResponse(BaseModel):
    calibration_params: Dict[str, Any]
    academic_whitelist: List[str]
    model_config: Dict[str, Any]
    export_timestamp: float

# Dependency to get config manager
def get_config_manager():
    """Dependency to get scoring configuration manager"""
    try:
        from app.services.scoring.scoring_config import get_scoring_config
        return get_scoring_config()
    except ImportError:
        raise HTTPException(status_code=503, detail="Configuration management not available")

@router.post("/calibration/update")
async def update_calibration(
    request: CalibrationRequest,
    config_manager = Depends(get_config_manager)
):
    """
    Update calibration parameters based on real scoring data
    
    Example payload:
    {
        "calibration_data": [
            {
                "raw_scores": {"content": 4, "grammar": 1.5, "spelling": 2, ...},
                "expected_score": 65
            }
        ],
        "description": "APEUni benchmark data 2024-01"
    }
    """
    try:
        # Get enhanced scorer for calibration
        from app.services.scoring.enhanced_write_essay_scorer import get_enhanced_essay_scorer
        scorer = get_enhanced_essay_scorer()
        
        # Convert request data to calibration format
        calibration_data = [
            {
                "raw_scores": item.raw_scores,
                "expected_score": item.expected_score
            }
            for item in request.calibration_data
        ]
        
        # Perform calibration
        updated_params = scorer.calibrate_score_mapping(calibration_data)
        
        # Save to persistent storage
        success = config_manager.save_calibration_params(updated_params)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save calibration parameters")
        
        return {
            "success": True,
            "message": f"Calibration updated with {len(calibration_data)} data points",
            "updated_parameters": updated_params,
            "description": request.description,
            "calibration_timestamp": time.time()
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Enhanced scorer not available for calibration")
    except Exception as e:
        logger.error(f"Calibration update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

@router.get("/calibration/current")
async def get_current_calibration(config_manager = Depends(get_config_manager)):
    """Get current calibration parameters"""
    try:
        params = config_manager.get_calibration_params()
        return {
            "success": True,
            "calibration_params": params
        }
    except Exception as e:
        logger.error(f"Failed to get calibration params: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/academic-whitelist/update")
async def update_academic_whitelist(
    request: AcademicWhitelistRequest,
    config_manager = Depends(get_config_manager)
):
    """
    Add or remove terms from academic whitelist
    
    Example payload:
    {
        "terms": ["nanotechnology", "blockchain", "genomics"],
        "action": "add"
    }
    """
    try:
        if request.action == "add":
            success = config_manager.add_academic_terms(request.terms)
            message = f"Added {len(request.terms)} terms to academic whitelist"
        elif request.action == "remove":
            success = config_manager.remove_academic_terms(request.terms)
            message = f"Removed {len(request.terms)} terms from academic whitelist"
        else:
            raise HTTPException(status_code=400, detail="Action must be 'add' or 'remove'")
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to {request.action} terms")
        
        updated_whitelist = config_manager.get_academic_whitelist()
        
        return {
            "success": True,
            "message": message,
            "terms_affected": request.terms,
            "total_whitelist_size": len(updated_whitelist),
            "action": request.action
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Academic whitelist update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/academic-whitelist/current")
async def get_academic_whitelist(config_manager = Depends(get_config_manager)):
    """Get current academic whitelist"""
    try:
        whitelist = config_manager.get_academic_whitelist()
        return {
            "success": True,
            "academic_whitelist": sorted(whitelist),
            "total_terms": len(whitelist)
        }
    except Exception as e:
        logger.error(f"Failed to get academic whitelist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model-config/update")
async def update_model_config(
    request: ModelConfigRequest,
    config_manager = Depends(get_config_manager)
):
    """
    Update model configuration
    
    Example payload:
    {
        "config": {
            "batch_size": 5,
            "cache_ttl_seconds": 7200,
            "gpu_enabled": true
        }
    }
    """
    try:
        success = config_manager.update_model_config(request.config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update model configuration")
        
        updated_config = config_manager.get_model_config()
        
        return {
            "success": True,
            "message": "Model configuration updated successfully",
            "updated_config": updated_config,
            "changes_applied": request.config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model config update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-config/current")
async def get_model_config(config_manager = Depends(get_config_manager)):
    """Get current model configuration"""
    try:
        config = config_manager.get_model_config()
        return {
            "success": True,
            "model_config": config
        }
    except Exception as e:
        logger.error(f"Failed to get model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export", response_model=ConfigExportResponse)
async def export_all_config(config_manager = Depends(get_config_manager)):
    """Export all configuration for backup"""
    try:
        config_data = config_manager.export_config()
        return ConfigExportResponse(**config_data)
    except Exception as e:
        logger.error(f"Config export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import")
async def import_config(
    config_data: ConfigExportResponse,
    config_manager = Depends(get_config_manager)
):
    """Import configuration from backup"""
    try:
        success = config_manager.import_config(config_data.dict())
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to import configuration")
        
        return {
            "success": True,
            "message": "Configuration imported successfully",
            "import_timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Config import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats():
    """Get parsing cache statistics"""
    try:
        from app.services.scoring.parsing_cache import get_parsing_cache
        cache = get_parsing_cache()
        
        stats = cache.get_cache_stats()
        
        return {
            "success": True,
            "cache_stats": stats
        }
        
    except ImportError:
        return {
            "success": False,
            "message": "Caching system not available"
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear")
async def clear_cache(keep_persistent: bool = False):
    """Clear parsing cache"""
    try:
        from app.services.scoring.parsing_cache import get_parsing_cache
        cache = get_parsing_cache()
        
        success = cache.clear_cache(keep_persistent=keep_persistent)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "kept_persistent": keep_persistent
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Caching system not available")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/optimize")
async def optimize_cache(typical_word_count: int = 250):
    """Optimize cache settings for typical essay length"""
    try:
        from app.services.scoring.parsing_cache import get_parsing_cache
        cache = get_parsing_cache()
        
        cache.optimize_cache_for_essay_length(typical_word_count)
        
        return {
            "success": True,
            "message": f"Cache optimized for essays ~{typical_word_count} words",
            "typical_word_count": typical_word_count
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Caching system not available")
    except Exception as e:
        logger.error(f"Failed to optimize cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health")
async def get_system_health():
    """Get comprehensive system health check"""
    try:
        health_info = {
            "timestamp": time.time(),
            "config_manager": True,
            "enhanced_scorer": False,
            "caching_system": False,
            "gpt_available": False,
            "models_loaded": {}
        }
        
        # Check enhanced scorer
        try:
            from app.services.scoring.enhanced_write_essay_scorer import get_enhanced_essay_scorer
            scorer = get_enhanced_essay_scorer()
            health_info["enhanced_scorer"] = True
            health_info["gpt_available"] = scorer.use_gpt
            health_info["models_loaded"] = {
                "gector": scorer.gector_model is not None,
                "language_tool": scorer.language_tool is not None,
                "sentence_transformer": scorer.sentence_model is not None,
                "hunspell": scorer.hunspell_checker is not None
            }
        except:
            pass
        
        # Check caching system
        try:
            from app.services.scoring.parsing_cache import get_parsing_cache
            cache = get_parsing_cache()
            health_info["caching_system"] = True
            health_info["cache_stats"] = cache.get_cache_stats()
        except:
            pass
        
        # Check configuration
        try:
            config_manager = get_config_manager()
            health_info["calibration_params"] = config_manager.get_calibration_params()
            health_info["academic_whitelist_size"] = len(config_manager.get_academic_whitelist())
        except:
            health_info["config_manager"] = False
        
        return {
            "success": True,
            "health": health_info
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress-test/run")
async def run_stress_test(num_tests: int = 20, max_workers: int = 5):
    """Run a quick stress test (async operation)"""
    try:
        # Import stress tester
        import asyncio
        import concurrent.futures
        
        # Simple stress test
        from app.services.scoring.enhanced_write_essay_scorer import score_enhanced_write_essay
        
        test_essay = """
Technology has transformed modern education significantly. Online learning platforms provide access to courses worldwide, enabling students to learn from expert instructors regardless of geographical location.

However, technology also presents challenges. Digital divide issues mean not all students have equal access to devices and internet connectivity. This can exacerbate educational inequalities rather than reducing them.

Despite these challenges, technology's benefits in education outweigh the drawbacks when implemented thoughtfully. Blended learning approaches that combine traditional teaching with digital tools often produce the best outcomes for student engagement and achievement.
"""
        
        test_prompt = "Discuss the impact of technology on education"
        
        start_time = time.time()
        
        # Run parallel tests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(score_enhanced_write_essay, test_essay, test_prompt)
                for _ in range(num_tests)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # Calculate results
        successful = sum(1 for r in results if r.get('success', False))
        total_time = end_time - start_time
        
        return {
            "success": True,
            "stress_test_results": {
                "total_tests": num_tests,
                "successful_tests": successful,
                "success_rate": successful / num_tests,
                "total_time": total_time,
                "throughput_per_second": num_tests / total_time,
                "average_time_per_test": total_time / num_tests
            }
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Enhanced scorer not available for stress testing")
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))