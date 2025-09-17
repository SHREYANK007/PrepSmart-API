#!/usr/bin/env python3
"""
Scoring Configuration Management
Handles persistent storage of calibration parameters and academic whitelists
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class ScoringConfigManager:
    """
    Manages persistent configuration for scoring system
    - Calibration parameters
    - Academic whitelists
    - Model configurations
    """
    
    def __init__(self, config_dir: str = "scoring_config"):
        """Initialize config manager with persistent storage"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Config file paths
        self.calibration_file = self.config_dir / "calibration_params.json"
        self.academic_whitelist_file = self.config_dir / "academic_whitelist.json"
        self.model_config_file = self.config_dir / "model_config.json"
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Load configurations
        self._load_all_configs()
        
        logger.info(f"✅ ScoringConfigManager initialized with config dir: {self.config_dir}")
    
    def _load_all_configs(self):
        """Load all configuration files"""
        self.calibration_params = self._load_calibration_params()
        self.academic_whitelist = self._load_academic_whitelist()
        self.model_config = self._load_model_config()
    
    def _load_calibration_params(self) -> Dict[str, Any]:
        """Load calibration parameters from storage"""
        default_params = {
            'scale_factor': 3.46,
            'curve_type': 'sigmoid',
            'sigmoid_steepness': 0.15,
            'sigmoid_midpoint': 13.0,
            'logarithmic_base': 1.5,
            'min_score': 10,
            'max_score': 90,
            'last_calibrated': None,
            'calibration_samples': 0
        }
        
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r', encoding='utf-8') as f:
                    loaded_params = json.load(f)
                    # Merge with defaults to handle new parameters
                    default_params.update(loaded_params)
                    logger.info("✅ Loaded calibration parameters from storage")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load calibration params: {e}")
        
        return default_params
    
    def _load_academic_whitelist(self) -> List[str]:
        """Load academic whitelist from storage"""
        default_whitelist = [
            'methodology', 'epistemology', 'paradigm', 'heuristic', 'meta-analysis',
            'quasi-experimental', 'socioeconomic', 'interdisciplinary', 'counterargument',
            'globalization', 'digitalization', 'urbanization', 'industrialization',
            'sustainability', 'biodiversity', 'cryptocurrency', 'nanotechnology',
            'anthropocene', 'blockchain', 'genomics', 'proteomics', 'bioinformatics',
            'neuroscience', 'cybersecurity', 'algorithmic', 'phenomenological',
            'ethnographic', 'longitudinal', 'cross-sectional', 'randomized',
            'placebo-controlled', 'meta-cognitive', 'socio-cultural', 'post-colonial',
            'intersectionality', 'hegemonic', 'dialectical', 'hermeneutic'
        ]
        
        if self.academic_whitelist_file.exists():
            try:
                with open(self.academic_whitelist_file, 'r', encoding='utf-8') as f:
                    loaded_whitelist = json.load(f)
                    # Merge with defaults and remove duplicates
                    combined = list(set(default_whitelist + loaded_whitelist))
                    logger.info(f"✅ Loaded {len(combined)} academic whitelist terms")
                    return combined
            except Exception as e:
                logger.warning(f"⚠️ Failed to load academic whitelist: {e}")
        
        return default_whitelist
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from storage"""
        default_config = {
            'gector_model': 'vennify/t5-base-grammar-correction',
            'sentence_transformer': 'all-MiniLM-L6-v2',
            'batch_size': 3,
            'max_sequence_length': 512,
            'gpu_enabled': True,
            'cache_parsed_results': True,
            'cache_ttl_seconds': 3600,  # 1 hour
            'upgrade_path': {
                'next_gector_model': 'roberta-gec',
                'upgrade_priority': 'medium',
                'upgrade_notes': 'Consider upgrading when infrastructure allows'
            }
        }
        
        if self.model_config_file.exists():
            try:
                with open(self.model_config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info("✅ Loaded model configuration from storage")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load model config: {e}")
        
        return default_config
    
    def save_calibration_params(self, params: Dict[str, Any]) -> bool:
        """Save calibration parameters to persistent storage"""
        with self._lock:
            try:
                # Update timestamp
                import time
                params['last_calibrated'] = time.time()
                
                # Save to file
                with open(self.calibration_file, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)
                
                # Update in-memory
                self.calibration_params = params
                
                logger.info("✅ Calibration parameters saved successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to save calibration params: {e}")
                return False
    
    def add_academic_terms(self, terms: List[str]) -> bool:
        """Add terms to academic whitelist"""
        with self._lock:
            try:
                # Add new terms and remove duplicates
                updated_whitelist = list(set(self.academic_whitelist + terms))
                
                # Save to file
                with open(self.academic_whitelist_file, 'w', encoding='utf-8') as f:
                    json.dump(sorted(updated_whitelist), f, indent=2, ensure_ascii=False)
                
                # Update in-memory
                self.academic_whitelist = updated_whitelist
                
                logger.info(f"✅ Added {len(terms)} terms to academic whitelist")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to update academic whitelist: {e}")
                return False
    
    def remove_academic_terms(self, terms: List[str]) -> bool:
        """Remove terms from academic whitelist"""
        with self._lock:
            try:
                # Remove terms
                updated_whitelist = [term for term in self.academic_whitelist if term not in terms]
                
                # Save to file
                with open(self.academic_whitelist_file, 'w', encoding='utf-8') as f:
                    json.dump(sorted(updated_whitelist), f, indent=2, ensure_ascii=False)
                
                # Update in-memory
                self.academic_whitelist = updated_whitelist
                
                logger.info(f"✅ Removed {len(terms)} terms from academic whitelist")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to remove terms from academic whitelist: {e}")
                return False
    
    def update_model_config(self, config: Dict[str, Any]) -> bool:
        """Update model configuration"""
        with self._lock:
            try:
                # Merge with existing config
                self.model_config.update(config)
                
                # Save to file
                with open(self.model_config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.model_config, f, indent=2, ensure_ascii=False)
                
                logger.info("✅ Model configuration updated successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to update model config: {e}")
                return False
    
    def get_calibration_params(self) -> Dict[str, Any]:
        """Get current calibration parameters"""
        return self.calibration_params.copy()
    
    def get_academic_whitelist(self) -> List[str]:
        """Get current academic whitelist"""
        return self.academic_whitelist.copy()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get current model configuration"""
        return self.model_config.copy()
    
    def export_config(self) -> Dict[str, Any]:
        """Export all configuration for backup"""
        return {
            'calibration_params': self.calibration_params,
            'academic_whitelist': self.academic_whitelist,
            'model_config': self.model_config,
            'export_timestamp': time.time()
        }
    
    def import_config(self, config: Dict[str, Any]) -> bool:
        """Import configuration from backup"""
        try:
            if 'calibration_params' in config:
                self.save_calibration_params(config['calibration_params'])
            
            if 'academic_whitelist' in config:
                with open(self.academic_whitelist_file, 'w', encoding='utf-8') as f:
                    json.dump(config['academic_whitelist'], f, indent=2, ensure_ascii=False)
                self.academic_whitelist = config['academic_whitelist']
            
            if 'model_config' in config:
                self.update_model_config(config['model_config'])
            
            logger.info("✅ Configuration imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to import configuration: {e}")
            return False

# Global instance
_global_config_manager = None

def get_scoring_config() -> ScoringConfigManager:
    """Get or create global scoring configuration manager"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ScoringConfigManager()
    return _global_config_manager