import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import React, { useState, useEffect } from 'react';
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";
import { Box, TextField, Button } from '@mui/material';

export const MLFlowIcon = ({ size = "1rem", style = {} }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      height={size}
      width={size}
      style={style}
      viewBox="0 0 600 500"
    >
      <defs id="defsdoc">
        <pattern
          id="patternBool"
          x="0"
          y="0"
          width="10"
          height="10"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(35)"
        >
          <circle
            fill="white"
            cx="5"
            cy="5"
            r="4"
            style={{ stroke: 'none' }}
          ></circle>
        </pattern>
      </defs>
      <g id="fileImp-172351224" className="cosito">
        <path
          fill="white"
          id="pathImp-357635842"
          className="grouped"
          d="M248.3119 62.5338C103.5671 64.4669 15.1926 222.3822 89.2403 346.7802 93.3076 353.6133 97.8042 360.1809 102.7021 366.4439 102.7021 366.4439 159.9718 324.3778 159.9718 324.3778 102.9004 253.4665 143.9869 147.3579 233.9289 133.3832 239.5154 132.5145 245.1564 132.0492 250.81 131.9883 250.81 131.9883 250.81 176.7558 250.81 176.7558 250.81 176.7558 358.6668 96.4959 358.6668 96.4959 326.3675 73.8514 287.7527 61.967 248.3119 62.5338 248.3119 62.5338 248.3119 62.5338 248.3119 62.5338M397.6064 133.3155C397.6064 133.3155 340.3368 175.3973 340.3368 175.3973 397.4206 246.2993 356.3513 352.4141 266.4124 366.4043 260.815 367.2747 255.163 367.7415 249.4985 367.8024 249.4985 367.8024 249.4985 323.0349 249.4985 323.0349 249.4985 323.0349 141.6417 403.2948 141.6417 403.2948 260.292 486.2716 424.2661 409.6766 436.7943 265.4244 440.9115 218.0163 426.9079 170.8081 397.6064 133.3155 397.6064 133.3155 397.6064 133.3155 397.6064 133.3155"
        ></path>
      </g>
    </svg>
  );
};

function useServerAvailability(defaultUrl) {
  const [serverAvailable, setServerAvailable] = useState(true);
  const [url, setUrl] = useState(defaultUrl);

  useEffect(() => {
    fetch(url, { mode: 'no-cors' })
      .then(() => setServerAvailable(true))
      .catch(() => setServerAvailable(false));
  }, [url]);

  return { serverAvailable, setServerAvailable, url, setUrl };
}

const URLInputForm = ({ onSubmit }) => {
  const [inputUrl, setInputUrl] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(inputUrl);
  };

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        display: "flex",
        justifyContent: "space-between",
        p: 1,
        bgcolor: "background.paper",
        borderRadius: 1,
      }}
      noValidate
      autoComplete="off"
    >
      <TextField
        label="MLFlow Server URL"
        variant="outlined"
        size="small"
        value={inputUrl}
        onChange={(e) => setInputUrl(e.target.value)}
        sx={{ width: "80%" }}
      />
      <Button type="submit" variant="contained" sx={{ marginLeft: 2 }}>
        Update URL
      </Button>
    </Box>
  );
};


export default function MLFlowPanel() {
  const defaultUrl = "http://127.0.0.1:8080";
  const { serverAvailable, setServerAvailable, url, setUrl } = useServerAvailability(defaultUrl);

  const handleUpdateUrl = (newUrl) => {
    setUrl(newUrl);
  };

  const datasetName = useRecoilValue(fos.datasetName);
  console.log(datasetName);
  // Use this dataset name to get candidate experiment urls...


  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {!serverAvailable && <URLInputForm onSubmit={handleUpdateUrl} />}
      <iframe
        style={{
          flexGrow: 1,
          border: "none",
        }}
        src={url}
        title="MLFlow Embedded"
        allowFullScreen
      ></iframe>
    </Box>
  );
}


registerComponent({
  name: "MLFlowPanel",
  label: "MLFlow Dashboard",
  component: MLFlowPanel,
  type: PluginComponentType.Panel,
  Icon: () => <MLFlowIcon size={"1rem"} style={{ marginRight: "0.5rem" }} />,
});
