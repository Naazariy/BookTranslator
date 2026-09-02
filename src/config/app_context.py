from dependency_injector import containers, providers
import logging

from src.config.settings import Settings
from src.document_manager.manager import DocumentManager
from src.knowledge_base.sqlite_repository import SQLiteKnowledgeBaseRepository
from src.chunking.manager import ChunkManager
from src.translation.pipeline import TwoStageTranslationPipeline
from src.translation.nllb_engine import NLLBMachineTranslationEngine
from src.translation.aya_editing_engine import AyaEditingEngine
from src.preprocessing.pipeline import PreprocessingPipeline
from src.quality.pipeline import QualityPipeline


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Repositories
    kb_repository = providers.Singleton(
        SQLiteKnowledgeBaseRepository,
        db_path=config.db_path
    )
    
    # Engines
    nllb_engine = providers.Singleton(
        NLLBMachineTranslationEngine,
        model_name=config.nllb_model_name,
        device=config.device
    )
    
    aya_engine = providers.Singleton(
        AyaEditingEngine,
        model_name=config.aya_model_name,
        device=config.device,
        temperature=config.aya_temperature,
        top_p=config.aya_top_p,
        repetition_penalty=config.aya_repetition_penalty,
        seed=config.aya_seed
    )
    
    # Services
    document_manager = providers.Singleton(DocumentManager)
    
    chunk_manager = providers.Singleton(ChunkManager)
    
    preprocessing_pipeline = providers.Singleton(
        PreprocessingPipeline,
        kb_repo=kb_repository
    )
    
    translation_pipeline = providers.Singleton(
        TwoStageTranslationPipeline,
        nllb_engine=nllb_engine,
        aya_engine=aya_engine,
        kb_repo=kb_repository
    )
    
    quality_pipeline = providers.Singleton(QualityPipeline)
