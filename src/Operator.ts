import {
  Operator,
  OperatorConfig,
  ExecutionContext,
  registerOperator,
} from "@fiftyone/operators";
import { useSetRecoilState } from "recoil";
import { iframeURLAtom } from "./State";

class SetIframeURL extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "set_iframe_url",
      label: "SetIframeURL",
      unlisted: true,
    });
  }
  useHooks(): {} {
    const setIframeUrl = useSetRecoilState(iframeURLAtom);
    return {
      setIframeUrl: setIframeUrl,
    };
  }
  async execute({ hooks, params }: ExecutionContext) {
    hooks.setIframeUrl(params.url);
  }
}

registerOperator(SetIframeURL, "@voxel51/mlflow");
