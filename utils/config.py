from utils.compute import QuantMode, QuantMethod, RepackMethod


class ExtractCacheConfig:
    def __init__(self, collect_round: int):
        self.collect_round = collect_round
        self.key_caches = {}
        self.value_caches = {}

    def size(self):
        total_size = 0
        for round_ks in self.key_caches.values():
            for k in round_ks.values():
                total_size += k.numel() * 16
        
        for round_vs in self.value_caches.values():
            for v in round_vs.values():
                total_size += v.numel() * 16

        return total_size

class PackKVCacheConfig:
    def __init__(
            self,
            model_name: str,
            quant_method: QuantMethod,
            repack_method: RepackMethod,
            high_precision_zero_point: bool,
            block_size: int,
            buffer_size: int,
            pack_size: int,
            k_quant_scale_rel: float,
            v_quant_scale_rel: float,
            # enable_k_minus_avg: bool,
            enable_quant: bool = True
    ):
        self.enable_quant: bool = enable_quant
        self.model_name: str = model_name
        self.quant_method: QuantMethod = quant_method
        self.repack_method: RepackMethod = repack_method
        self.high_precision_zero_point: bool = high_precision_zero_point
        self.block_size: int = block_size
        self.buffer_size: int = buffer_size
        self.pack_size: int = pack_size
        # self.enable_k_minus_avg: bool = enable_k_minus_avg
        self.k_quant_scale_rel: float = k_quant_scale_rel
        self.v_quant_scale_rel: float = v_quant_scale_rel

    # to string print
    def __str__(self):
        # parse class as json
        json_ = {
            "enable_quant": self.enable_quant,
            "model_name": self.model_name,
        }
        if self.enable_quant:
            # json_["enable_k_minus_avg"] = self.enable_k_minus_avg
            json_["quant_method"] = self.quant_method.value
            json_["repack_method"] = self.repack_method.value
            json_["high_precision_zero_point"] = self.high_precision_zero_point
            json_["block_size"] = self.block_size
            json_["buffer_size"] = self.buffer_size
            json_["pack_size"] = self.pack_size
            json_["k_quant_scale_rel"] = self.k_quant_scale_rel
            json_["v_quant_scale_rel"] = self.v_quant_scale_rel

        return str(json_)

    @staticmethod
    def from_str(json_str: str):
        # parse json string to class
        json_ = eval(json_str)
        if json_["enable_quant"]:
            quant_method = QuantMethod(json_["quant_method"])
            repack_method = RepackMethod(json_["repack_method"])
            high_precision_zero_point = json_["high_precision_zero_point"]
            block_size = json_["block_size"]
            buffer_size = json_["buffer_size"]
            pack_size = json_["pack_size"]
            k_quant_scale_rel = json_["k_quant_scale_rel"]
            v_quant_scale_rel = json_["v_quant_scale_rel"]
        else:
            quant_method = None
            repack_method = None
            block_size = None
            buffer_size = None
            pack_size = None
            k_quant_scale_rel = None
            v_quant_scale_rel = None

        return PackKVCacheConfig(
            model_name=json_["model_name"],
            enable_quant=json_["enable_quant"],
            # enable_k_minus_avg=json_["enable_k_minus_avg"],
            quant_method=quant_method,
            repack_method=repack_method,
            high_precision_zero_point=high_precision_zero_point,
            block_size=block_size,
            buffer_size=buffer_size,
            pack_size=pack_size,
            k_quant_scale_rel=k_quant_scale_rel,
            v_quant_scale_rel=v_quant_scale_rel,
        )

    def __eq__(self, other):
        if not isinstance(other, PackKVCacheConfig):
            return False
        if self.enable_quant != other.enable_quant:
            return False
        if self.model_name != other.model_name:
            return False
        # if self.enable_k_minus_avg != other.enable_k_minus_avg:
        #     return False
        if self.quant_method != other.quant_method:
            return False
        if self.repack_method != other.repack_method:
            return False
        if self.high_precision_zero_point != other.high_precision_zero_point:
            return False
        if self.block_size != other.block_size:
            return False
        if self.buffer_size != other.buffer_size:
            return False
        if self.pack_size != other.pack_size:
            return False
        if self.k_quant_scale_rel != other.k_quant_scale_rel:
            return False
        if self.v_quant_scale_rel != other.v_quant_scale_rel:
            return False
        return True

    def __hash__(self):
        return hash((self.enable_quant, self.model_name,
                     # self.enable_k_minus_avg,
                     self.quant_method, self.repack_method, self.high_precision_zero_point, self.block_size, self.buffer_size,
                     self.pack_size,
                     self.k_quant_scale_rel, self.v_quant_scale_rel))
    
    # for print
    def __repr__(self):
        return self.__str__()
