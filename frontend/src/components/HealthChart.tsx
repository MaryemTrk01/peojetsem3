import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type Point = { [key: string]: number };

export default function HealthChart({
  data,
  dataKey,
  label,
  color = "#3b82f6",
}: {
  data: Point[];
  dataKey: string;
  label: string;
  color?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{ top: 5, right: 10, left: -20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="t"
          tick={{ fontSize: 12, fill: "#94a3b8" }}
          stroke="#475569"
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 12, fill: "#94a3b8" }}
          stroke="#475569"
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #475569",
            borderRadius: "8px",
          }}
          labelStyle={{ color: "#e2e8f0" }}
        />
        <Line
          type="monotone"
          dataKey={dataKey}
          dot={false}
          strokeWidth={2}
          stroke={color}
          isAnimationActive={true}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
