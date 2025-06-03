
interface CounterDisplayProps {
  count: number;
  label: string;
  size?: 'normal' | 'large';
}

export const CounterDisplay = ({ count, label, size = 'normal' }: CounterDisplayProps) => {
  const numberSize = size === 'large' ? 'text-6xl md:text-8xl' : 'text-4xl md:text-5xl';
  const labelSize = size === 'large' ? 'text-xl' : 'text-lg';

  return (
    <div className="text-center">
      <div className={`${numberSize} font-bold text-yellow-400 mb-2 animate-pulse`}>
        {count.toLocaleString('tr-TR')}
      </div>
      <div className={`${labelSize} text-gray-600`}>
        {label}
      </div>
    </div>
  );
};
