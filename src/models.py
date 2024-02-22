import json
import requests
import asyncio
import time
import io

from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence

from typing_extensions import Self

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.components.sensor import Sensor
from viam import logging
from viam.resource.types import Model, ModelFamily
from sklearn.ensemble import RandomForestClassifier

LOGGER = logging.getLogger(__name__)

class PrusaConnectCameraServer(Sensor):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "showtell"), "example")
    clf: RandomForestClassifier = None
    X = [[ 1,  2,  3],  # 2 samples, 3 features
        [11, 12, 13]]
    y = [0, 1]  # classes of each sample


    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        snapshotter = cls(config.name)
        snapshotter.reconfigure(config, dependencies)
        return snapshotter
    
    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        return None
    
    def reconfigure(self,
                    config: ComponentConfig,
                    dependencies: Mapping[ResourceName, ResourceBase]):
        self.clf = RandomForestClassifier(random_state=0)
        self.clf.fit(self.X, self.y)

    async def get_readings(self, extra: Optional[Dict[str, Any]] = None, **kwargs) -> Mapping[str, Any]:
        return {"predictions": str(self.clf.predict(self.X))}