import { useEffect, useRef, type RefObject } from 'react'
import { Clip, Close, Send } from './icons'

export function Composer({
  formRef,
  textareaRef,
  value,
  onChange,
  file,
  onFile,
  onSubmit,
  busy,
  placeholder = 'Hỏi về một bài báo hoặc chủ đề…',
}: {
  formRef: RefObject<HTMLFormElement>
  textareaRef: RefObject<HTMLTextAreaElement>
  value: string
  onChange: (value: string) => void
  file: File | null
  onFile: (file: File | null) => void
  onSubmit: () => void
  busy: boolean
  placeholder?: string
}) {
  const picker = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value, textareaRef])

  const canSend = !busy && (value.trim().length > 0 || !!file)
  return (
    <form
      ref={formRef}
      className="composer"
      onSubmit={(e) => {
        e.preventDefault()
        if (canSend) onSubmit()
      }}
    >
      <label htmlFor="q" className="sr">
        Hỏi về một bài báo hoặc chủ đề
      </label>
      <textarea
        id="q"
        ref={textareaRef}
        rows={1}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            if (canSend) onSubmit()
          }
        }}
      />
      <div className="row">
        <input
          ref={picker}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => {
            onFile(e.target.files?.[0] ?? null)
            e.target.value = ''
          }}
        />
        <button type="button" className="icon-btn" aria-label="Đính kèm PDF" onClick={() => picker.current?.click()}>
          <Clip />
        </button>
        {file && (
          <span className="attached">
            {file.name}
            <button type="button" aria-label="Bỏ file đính kèm" onClick={() => onFile(null)}>
              <Close />
            </button>
          </span>
        )}
        <button className="send" type="submit" aria-label="Gửi" disabled={!canSend}>
          <Send />
        </button>
      </div>
    </form>
  )
}
