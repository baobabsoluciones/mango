from mango.config import BaseConfig, ConfigParameter
from mango.models.genetic.const import INDIVIDUAL_ENCODINGS


class GeneticBaseConfig(BaseConfig):
    __params = {
        "main": [
            ConfigParameter("population_size", int, 100),
            ConfigParameter("max_generations", int, 500),
            ConfigParameter("fitness_threshold", float),
            ConfigParameter("mutation_rate", float, 0.1),
            ConfigParameter("max_stagnation", int, 50),
            ConfigParameter("verbose", bool, False),
            ConfigParameter("save_log", bool, False),
            ConfigParameter("save_backups", bool, False),
            ConfigParameter("backup_population_frequency", int, 100),
            ConfigParameter("backup_individual_frequency", int, 10),
            ConfigParameter("import_from_backup", bool, False),
            ConfigParameter("backup_import_path", str, ""),
            ConfigParameter("backup_export_path", str, ""),
            ConfigParameter("exec_name", str, ""),
        ],
        "individual": [
            ConfigParameter("individual_encoding", str, validate=INDIVIDUAL_ENCODINGS),
            ConfigParameter("individual_values", list, secondary_type=float),
            ConfigParameter("individual_gene_min_value", float),
            ConfigParameter("individual_gene_max_value", float),
        ],
    }
